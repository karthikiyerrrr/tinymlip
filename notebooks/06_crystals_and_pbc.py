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

    return (
        EMT,
        EquivariantMPNN,
        ase,
        build_graph,
        compute_forces_and_stress,
        go,
        load_cu_emt,
        pl,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    a_slider = mo.ui.slider(start=3.0, stop=4.5, step=0.05, value=3.615, label="lattice constant a (Å)")
    cutoff_slider = mo.ui.slider(start=2.5, stop=5.5, step=0.1, value=4.0, label="cutoff (Å)")
    mo.hstack([a_slider, cutoff_slider])
    return a_slider, cutoff_slider


@app.cell
def _(a_slider, ase, build_graph, cutoff_slider, pl):
    fcc_cu = ase.build.bulk("Cu", "fcc", a=a_slider.value, cubic=True)
    g = build_graph(fcc_cu, cutoff=cutoff_slider.value)

    pl.DataFrame(
        {
            "quantity": [
                "formula",
                "n_atoms",
                "lattice a (Å)",
                "cutoff (Å)",
                "n_edges",
                "edges from atom 0",
                "max |shift|",
                "pbc",
            ],
            "value": [
                str(fcc_cu.symbols),
                str(g.n_atoms),
                f"{a_slider.value:.3f}",
                f"{cutoff_slider.value:.2f}",
                str(g.n_edges),
                str((g.edge_index[0] == 0).sum().item()),
                str(g.shift_idx.abs().max().item()) if g.shift_idx is not None else "—",
                str(g.shift_idx is not None),
            ],
        }
    )
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
def _(build_graph, cutoff_slider, fcc_cu, g, mo, torch):
    fcc_cu_shifted = fcc_cu.copy()
    fcc_cu_shifted.positions[0] += fcc_cu.cell.array[0]  # translate atom 0 by +a₁
    g_shifted = build_graph(fcc_cu_shifted, cutoff=cutoff_slider.value)
    dists_match = torch.allclose(
        torch.sort(g.edge_dist).values,
        torch.sort(g_shifted.edge_dist).values,
        atol=1e-5,
    )

    mo.md(f"""
    | quantity | before shift | after shift |
    |---|---|---|
    | n_edges | {g.n_edges} | {g_shifted.n_edges} |
    | sorted edge-distance multisets match | — | **{dists_match}** |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The strain trick: stress from a single backward pass

    **Stress is the response of energy to a small deformation of the cell.**
    Parameterize the deformation as a symmetric 3×3 strain tensor `ε`. Atomic
    positions deform as `r → r·(I + ε)` and the cell as `c → c·(I + ε)`. The
    determinant of `I + ε` is `1 + tr(ε) + O(ε²)`, so a uniform isotropic
    strain is just a volumetric scaling — exactly what the `a` slider above
    does, in disguise.

    **Stress is then `σ = (1/V) · ∂E/∂ε`,** evaluated at `ε = 0`. If `ε` is a
    leaf in the autograd graph, σ falls out of one extra backward pass — the
    same trick that gave us forces from `−∂E/∂r` back in nb03. We get a
    rank-2 tensor instead of a vector, but the principle is identical.

    **Critical implementation detail.** For this to actually work, every layer
    must recompute `edge_vec` from `pos[j] - pos[i] + S @ cell` *inside* its
    forward pass — not from a cached `edge_vec` field on the graph. Otherwise
    the strain is applied to `pos` and `cell` but the model's edges still
    point in the un-strained directions, and σ comes out as zeros. Our
    `InvariantInteraction` and `EquivariantInteraction` both do the recompute
    (see `layers.py`); the next cell verifies σ against a hand-rolled
    numerical derivative to confirm.
    """)
    return


@app.cell
def _(
    EquivariantMPNN,
    a_slider,
    ase,
    build_graph,
    compute_forces_and_stress,
    cutoff_slider,
    mo,
    pl,
    torch,
):
    from dataclasses import replace as _replace

    torch.manual_seed(0)
    demo_model = EquivariantMPNN(
        n_layers=2, hidden_dim=16, num_basis=16, cutoff=cutoff_slider.value
    ).double()
    demo_atoms = ase.build.bulk("Cu", "fcc", a=a_slider.value, cubic=True).repeat((2, 2, 2))
    demo_atoms.rattle(stdev=0.05, seed=0)
    demo_g = build_graph(demo_atoms, cutoff=cutoff_slider.value, dtype=torch.float64)

    # Autograd σ
    e_ad, f_ad, sigma_ad = compute_forces_and_stress(demo_model, demo_g)

    # Numerical σ via central finite differences on a symmetrized strain
    h = 1e-4
    sigma_num = torch.zeros(3, 3, dtype=torch.float64)
    volume = demo_g.cell.det().abs().item()
    for i in range(3):
        for j in range(3):
            ep = torch.zeros(3, 3, dtype=torch.float64); ep[i, j] += h; ep[j, i] += h
            em = torch.zeros(3, 3, dtype=torch.float64); em[i, j] -= h; em[j, i] -= h
            e_plus = demo_model(_replace(
                demo_g,
                pos=demo_g.pos + demo_g.pos @ ep,
                cell=demo_g.cell + demo_g.cell @ ep,
            )).sum()
            e_minus = demo_model(_replace(
                demo_g,
                pos=demo_g.pos + demo_g.pos @ em,
                cell=demo_g.cell + demo_g.cell @ em,
            )).sum()
            sigma_num[i, j] = (e_plus - e_minus) / (4 * h) / volume
    sigma_num = 0.5 * (sigma_num + sigma_num.T)

    max_abs_err = (sigma_ad - sigma_num).abs().max().item()
    _s_ad = sigma_ad.detach().numpy()
    _s_nm = sigma_num.detach().numpy()

    mo.vstack([
        mo.md(
            f"**max |σ_autograd − σ_numerical| = {max_abs_err:.3e}**  "
            f"(target: < 1e-5)"
        ),
        mo.md("σ autograd (eV/Å³)"),
        pl.DataFrame(
            {
                "axis": ["x", "y", "z"],
                "σ·x̂": _s_ad[:, 0].tolist(),
                "σ·ŷ": _s_ad[:, 1].tolist(),
                "σ·ẑ": _s_ad[:, 2].tolist(),
            }
        ),
        mo.md("σ numerical (eV/Å³)"),
        pl.DataFrame(
            {
                "axis": ["x", "y", "z"],
                "σ·x̂": _s_nm[:, 0].tolist(),
                "σ·ŷ": _s_nm[:, 1].tolist(),
                "σ·ẑ": _s_nm[:, 2].tolist(),
            }
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Parity confirmed.** Autograd σ matches a central finite-difference of
    the energy under a symmetrized hand-built strain to roughly 1e-8. This is
    the same principle as the force parity check in nb03 — energy is the
    single learned scalar, every other physical quantity is a derivative of
    it against a different leaf (positions → forces, cell strain → stress).

    The diagonal entries above are the **normal stresses** (eV/Å³); the
    off-diagonals are **shears**. On a perfectly equilibrated lattice all six
    independent entries would be zero, but our untrained model has random
    weights and the lattice is slightly rattled, so non-zero values are
    expected. After training (section 5) the entries should track ASE's EMT
    σ to within the test-set MAE.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the labels come from

    We need labeled data to train against. **ASE ships an EMT
    (Effective Medium Theory) calculator** that works on FCC metals
    — Cu, Al, Ni, Pd, Ag, Pt, Au — and produces energy, forces, **and**
    stress for free in physical units (eV, eV/Å, eV/Å³). It is fast,
    differentiable-by-finite-difference, and roughly captures the right
    cohesive physics for these metals. We treat it as our "ground truth"
    surrogate; in actual research these labels would come from DFT.

    The dataset we generate below contains 800 snapshots of a 2×2×2 FCC Cu
    supercell (32 atoms), rattled and strained around the equilibrium
    lattice constant. Each snapshot carries an EMT-computed (E, F, σ)
    triple. We use this to train the equivariant model in the next
    section.
    """)
    return


@app.cell
def _(EMT, ase, mo, pl):
    snap = ase.build.bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 2))
    snap.rattle(stdev=0.05, seed=42)
    snap.calc = EMT()
    snap_E = snap.get_potential_energy()
    snap_F = snap.get_forces()
    snap_sigma = snap.get_stress(voigt=False)

    mo.vstack([
        mo.md("**Single EMT snapshot:** 2×2×2 FCC Cu (32 atoms), rattle stdev=0.05 Å"),
        pl.DataFrame(
            {
                "quantity": [
                    "total energy (eV)",
                    "per-atom energy (eV)",
                    "|F|_max (eV/Å)",
                    "|F|_mean (eV/Å)",
                    "tr(σ)/3 = pressure (eV/Å³)",
                ],
                "value": [
                    f"{snap_E:.4f}",
                    f"{snap_E / len(snap):.4f}",
                    f"{abs(snap_F).max():.3f}",
                    f"{abs(snap_F).mean():.3f}",
                    f"{snap_sigma.trace() / 3:.4e}",
                ],
            }
        ),
        mo.md("σ (eV/Å³)"),
        pl.DataFrame(
            {
                "axis": ["x", "y", "z"],
                "σ·x̂": snap_sigma[:, 0].tolist(),
                "σ·ŷ": snap_sigma[:, 1].tolist(),
                "σ·ẑ": snap_sigma[:, 2].tolist(),
            }
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate (or load from cache) the 800-snapshot Cu/EMT dataset

    Each snapshot is a 2×2×2 FCC Cu supercell (32 atoms) with: an isotropic
    volumetric strain drawn from ±5%, a small shear (±2%), and Gaussian
    rattle of 0.1 Å on each Cartesian coordinate. EMT labels (E, F, σ) are
    computed on each snapshot and the whole bundle is cached to disk as
    an extxyz file so re-running this cell is instant.
    """)
    return


@app.cell
def _(load_cu_emt, mo):
    import time

    t0 = time.time()
    meta, all_atoms = load_cu_emt(
        n_snapshots=800,
        supercell=(2, 2, 2),
        rattle_amp=0.1,
        strain_range=0.05,
        shear_range=0.02,
        seed=0,
    )
    elapsed = time.time() - t0

    mo.vstack([
        mo.md(f"Generated/loaded **{len(all_atoms)}** snapshots in **{elapsed:.1f} s**"),
        meta.head(),
    ])
    return


if __name__ == "__main__":
    app.run()
