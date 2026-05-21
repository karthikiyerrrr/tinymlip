import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 05 — Equivariant Model (PaiNN)

        **What this notebook teaches.** Why pure-scalar (invariant) hidden
        representations leave directional information on the table, and how a
        PaiNN-style equivariant interaction adds vector features per atom — `v`
        — that rotate with the molecule. You'll see live rotation equivariance,
        dissect the message phase, then train the equivariant model on rMD17 and
        compare it head-to-head against the invariant model from nb04.

        **Prerequisites.** You've worked through:
        - **nb02** (message passing on graphs, `index_add_`, `InvariantInteraction` walkthrough), and
        - **nb04** (training loop, energy + force loss, rMD17 loading).

        **By the end you can:** explain what `s` and `v` features are for, read
        PaiNN's message + update phases, verify equivariance live, and compare
        invariant vs equivariant training curves on the same dataset under
        identical hyperparameters.

        **Reference.** Schütt, Unke & Gastegger, *Equivariant message passing
        for the prediction of tensorial properties and molecular spectra*, ICML
        2021 (PaiNN). Deviations from the reference are documented in the
        `EquivariantInteraction` docstring.

        Forward link: **nb06** takes this to crystals with periodic boundary conditions.
        """
    )
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

    from tinymlip.data import load_rmd17, make_collate, to_torch_dataset
    from tinymlip.forces import compute_forces
    from tinymlip.graph import AtomGraph, build_graph
    from tinymlip.layers import EquivariantInteraction
    from tinymlip.models import EquivariantMPNN, InvariantMPNN
    from tinymlip.train import (
        apply_atomic_reference,
        fit_atomic_reference,
        train,
    )
    from tinymlip.viz import element_color, plot_graph_3d

    return (
        AtomGraph,
        DataLoader,
        EquivariantInteraction,
        EquivariantMPNN,
        InvariantMPNN,
        apply_atomic_reference,
        build_graph,
        compute_forces,
        element_color,
        fit_atomic_reference,
        go,
        load_rmd17,
        make_collate,
        np,
        pl,
        plot_graph_3d,
        to_torch_dataset,
        torch,
        train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. The rotation hook — why scalars aren't enough

        Take a small molecule (H₂O) and an untrained `InvariantMPNN`. As we
        rotate the molecule rigidly, the **forces** rotate with it (they're a
        vector quantity derived from $-\partial E / \partial \mathbf{r}$, and
        $\mathbf{r}$ is the autograd leaf). But the model's internal **hidden
        scalar features** don't change at all — they're invariant by
        construction.

        Drag the slider below. Watch the arrows rotate; watch the scalar bars
        stay flat. The arrows are the physics; the bars are everything our model
        "knows" about an atom internally. Notice the mismatch.

        **The fix:** add a vector channel `v` per atom that rotates *with* the
        molecule, so internal features can carry direction too.
        """
    )
    return


@app.cell
def _(InvariantMPNN, torch):
    from ase.build import molecule as ase_molecule

    water = ase_molecule("H2O")
    water_cutoff = 2.0  # H-O ~ 0.96 A, H-H ~ 1.5 A; 2.0 A includes both

    torch.manual_seed(0)
    hook_model = InvariantMPNN(
        hidden_dim=16, num_basis=8, cutoff=water_cutoff, n_layers=2
    )
    hook_model.eval()
    return ase_molecule, hook_model, water, water_cutoff


@app.cell(hide_code=True)
def _(mo):
    rotation_angle_deg = mo.ui.slider(
        start=0,
        stop=360,
        step=5,
        value=0,
        label="rotation angle (deg, around z-axis)",
        show_value=True,
    )
    rotation_angle_deg
    return (rotation_angle_deg,)


@app.cell(hide_code=True)
def _(
    build_graph,
    compute_forces,
    element_color,
    go,
    hook_model,
    mo,
    np,
    rotation_angle_deg,
    torch,
    water,
    water_cutoff,
):
    # Rotate H2O around z by the slider angle, run hook_model, collect forces
    # and a few hidden scalars. Plot side-by-side.
    _theta = float(rotation_angle_deg.value) * np.pi / 180.0
    _R = torch.tensor(
        [
            [np.cos(_theta), -np.sin(_theta), 0.0],
            [np.sin(_theta), np.cos(_theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    _water_rot = water.copy()
    _water_rot.set_positions(water.get_positions() @ _R.numpy().T)

    _graph = build_graph(_water_rot, cutoff=water_cutoff)
    _graph.pos.requires_grad_(True)
    _e = hook_model(_graph)
    _forces = compute_forces(_e, _graph.pos).detach().numpy()

    # Hidden scalars at atom 0: post-embedding features before any
    # interaction (already invariant — that's the visual point).
    with torch.no_grad():
        _x0 = hook_model.embed(_graph.z)[0].numpy()  # [F]

    _pos_np = _graph.pos.detach().numpy()
    _z_np = _graph.z.numpy()

    _arrow_scale = 1.0
    _atoms_trace = go.Scatter3d(
        x=_pos_np[:, 0],
        y=_pos_np[:, 1],
        z=_pos_np[:, 2],
        mode="markers",
        marker=dict(size=14, color=[element_color(int(z)) for z in _z_np]),
        name="atoms",
    )
    _arrow_traces = []
    for _k in range(_graph.n_atoms):
        _start = _pos_np[_k]
        _end = _start + _arrow_scale * _forces[_k]
        _arrow_traces.append(
            go.Scatter3d(
                x=[_start[0], _end[0]],
                y=[_start[1], _end[1]],
                z=[_start[2], _end[2]],
                mode="lines",
                line=dict(color="crimson", width=6),
                showlegend=False,
            )
        )

    _left = go.Figure([_atoms_trace, *_arrow_traces])
    _left.update_layout(
        title="Forces (rotate with the molecule)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=380,
    )

    _right = go.Figure(go.Bar(x=[f"ch{i}" for i in range(8)], y=_x0[:8]))
    _right.update_layout(
        title="Hidden scalars at atom 0 (unchanged by rotation)",
        yaxis=dict(range=[float(_x0[:8].min()) - 0.5, float(_x0[:8].max()) + 0.5]),
        height=380,
        margin=dict(l=40, r=10, t=40, b=40),
    )

    mo.hstack([_left, _right])
    return


if __name__ == "__main__":
    app.run()
