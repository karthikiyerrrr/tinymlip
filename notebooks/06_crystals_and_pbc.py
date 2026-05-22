import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 06 — Crystals and Periodic Boundary Conditions

    **What this notebook teaches.** What changes when atoms live in a periodic
    crystal instead of a finite molecule: how the graph carries integer lattice
    shifts so edges can cross the unit cell boundary; how `atoms.cell` enters the
    autograd graph so we can derive **stress** as a third conservative quantity
    (after energy and forces) from a single learned energy via the strain trick
    `σ = (1/V) ∂E/∂ε`; and how the equivariant model from nb05 trains on a
    synthetic FCC copper dataset labeled by ASE's built-in EMT calculator.

    **Prerequisites.** nb05 (equivariant model + autograd forces under batching).

    **By the end you can:** build a PBC graph, explain why `edge_vec` must be
    recomputed from `pos + S @ cell` inside the model's forward pass, derive
    stress via the strain trick and train an equivariant model on E + F + σ.

    **Reference.** The strain-derivative formulation here is the standard MLIP
    convention (NequIP, MACE, Allegro, SchNetPack-PaiNN all do this). ASE's
    EMT calculator (Jacobsen, Stoltze, Nørskov 1996) provides the labels.

    **Budget.** ~10 min CPU at `tiny`. Mention `small` / `default` configs in
    prose only.

    **Where this leads.** Real DFT data (e.g. MPtraj), e3nn-based equivariant
    models (NequIP / MACE / Allegro), and NPT molecular dynamics using the
    trained potential.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import torch
    from torch.utils.data import DataLoader

    import ase.build
    from ase.calculators.emt import EMT

    from tinymlip.data import (
        generate_cu_dataset,
        load_cu_emt,
        make_collate,
        to_torch_dataset_cu_emt,
    )
    from tinymlip.forces import compute_forces_and_stress
    from tinymlip.graph import build_graph, collate_graphs
    from tinymlip.layers import EquivariantInteraction
    from tinymlip.models import EquivariantMPNN
    from tinymlip.train import energy_force_loss, fit_atomic_reference, train
    from tinymlip.viz import e_v_curve

    return ase, build_graph, go, torch


@app.cell(hide_code=True)
def _(mo):
    a_slider = mo.ui.slider(start=3.0, stop=4.5, step=0.05, value=3.615, label="lattice constant a (Å)")
    cutoff_slider = mo.ui.slider(start=2.5, stop=5.5, step=0.1, value=4.0, label="cutoff (Å)")
    mo.hstack([a_slider, cutoff_slider])
    return a_slider, cutoff_slider


@app.cell
def _(a_slider, ase, build_graph, cutoff_slider):
    fcc_cu = ase.build.bulk("Cu", "fcc", a=a_slider.value, cubic=True)
    print(fcc_cu)
    g = build_graph(fcc_cu, cutoff=cutoff_slider.value)
    print(g)
    print("edges from central atom 0:", (g.edge_index[0] == 0).sum().item())
    print("max |shift|:", g.shift_idx.abs().max().item() if g.shift_idx is not None else None)
    return fcc_cu, g


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why we need integer lattice shifts

    Under PBC, every atom has copies in the 26 surrounding image cells (the
    3×3×3 tiling minus the center). An edge within the cutoff can connect to
    any of these images. Each edge stores an integer shift `S ∈ ℤ³` so the
    model knows which image of the neighbor was used:

    \[
        \mathbf{r}_{ij} = \mathbf{r}_j + S \cdot \mathbf{cell} - \mathbf{r}_i.
    \]

    Without that shift, edges that wrap around the boundary would have the
    wrong distance and the wrong direction. The plot below shows the central
    cell (solid) and its 8 in-plane image cells (faded); orange edges connect
    to a neighbor in a non-zero image (`S ≠ 0`).
    """)
    return


@app.cell(hide_code=True)
def _(fcc_cu, g, go):
    def _image_tile_plot(graph, atoms):
        cell = atoms.cell.array
        pos = atoms.positions
        a1, a2 = cell[0], cell[1]
        src = graph.edge_index[0].numpy()
        dst = graph.edge_index[1].numpy()
        S = graph.shift_idx.numpy()

        fig = go.Figure()

        # Draw the 3x3 in-plane image cells (i, j) with i,j in {-1, 0, 1}
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                origin = i * a1 + j * a2
                corners_x = [origin[0], origin[0] + a1[0], origin[0] + a1[0] + a2[0], origin[0] + a2[0], origin[0]]
                corners_y = [origin[1], origin[1] + a1[1], origin[1] + a1[1] + a2[1], origin[1] + a2[1], origin[1]]
                is_central = (i == 0 and j == 0)
                fig.add_trace(go.Scatter(
                    x=corners_x, y=corners_y, mode="lines",
                    line=dict(color="#444" if is_central else "#bbb", width=2 if is_central else 1),
                    hoverinfo="skip", showlegend=False,
                ))
                # Atoms in this tile
                xs = pos[:, 0] + origin[0]
                ys = pos[:, 1] + origin[1]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(size=14 if is_central else 9,
                                color="#b87333" if is_central else "#e7c9a8",
                                line=dict(color="#333", width=1)),
                    hoverinfo="text",
                    text=[f"atom {k} tile ({i},{j})" for k in range(len(pos))],
                    showlegend=False,
                ))

        # Edges (only those originating from the central cell, projected to xy).
        # Only draw edges whose shift lies in the xy plane (S_z == 0) so they show
        # up in this 2D projection without crossing the figure.
        in_xy_drawn_intra = False
        in_xy_drawn_inter = False
        for k in range(len(src)):
            s = S[k]
            if s[2] != 0:
                continue
            a = pos[src[k]]
            b = pos[dst[k]] + s[0] * a1 + s[1] * a2
            is_intra = (s == 0).all()
            color = "#1f77b4" if is_intra else "#ff7f0e"
            name = "intra-cell (S=0)" if is_intra else "cross-cell (S≠0)"
            show = (is_intra and not in_xy_drawn_intra) or (not is_intra and not in_xy_drawn_inter)
            if is_intra:
                in_xy_drawn_intra = True
            else:
                in_xy_drawn_inter = True
            fig.add_trace(go.Scatter(
                x=[a[0], b[0]], y=[a[1], b[1]], mode="lines",
                line=dict(color=color, width=1.2),
                name=name, legendgroup=name, showlegend=show,
                hoverinfo="skip",
            ))

        a = atoms.cell.array[0, 0]
        fig.update_layout(
            xaxis=dict(title="x (Å)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="y (Å)"),
            title=f"FCC Cu (a={a:.2f} Å), central cell + 8 image cells (z=0 slice)",
            width=600, height=600,
        )
        return fig


    _image_tile_plot(g, fcc_cu)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sanity check: lattice-translating one atom is a no-op

    If we slide atom 0 by one full lattice vector `+a₁`, the periodic image
    of that atom should land exactly where the original neighbor pattern
    expects it. The edge-distance multiset should be identical.
    """)
    return


@app.cell
def _(build_graph, cutoff_slider, fcc_cu, g, torch):
    fcc_cu_shifted = fcc_cu.copy()
    fcc_cu_shifted.positions[0] += fcc_cu.cell.array[0]  # translate atom 0 by +a₁
    g_shifted = build_graph(fcc_cu_shifted, cutoff=cutoff_slider.value)
    print("edges before:", g.n_edges, "edges after:", g_shifted.n_edges)
    print("sorted dists match:",
          torch.allclose(torch.sort(g.edge_dist).values,
                         torch.sort(g_shifted.edge_dist).values, atol=1e-5))
    return


if __name__ == "__main__":
    app.run()
