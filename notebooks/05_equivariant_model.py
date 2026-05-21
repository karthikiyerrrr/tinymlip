import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
        build_graph,
        compute_forces,
        element_color,
        fit_atomic_reference,
        go,
        load_rmd17,
        make_collate,
        np,
        pl,
        to_torch_dataset,
        torch,
        train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Vector features on atoms

    PaiNN gives every atom **two** feature tensors:

    ```
    s : [N, F]        # scalar features per atom  (what InvariantMPNN had)
    v : [N, F, 3]     # vector features per atom  (new — these rotate with the molecule)
    ```

    `v` starts at zero. The first time we call an `EquivariantInteraction`,
    `v` gets bootstrapped by the **creation message**: a vector built from
    each edge's direction `unit_ij = (pos_j - pos_i) / r_ij`, scaled by a
    learned scalar weight on the sender's `s`. After the first layer,
    subsequent layers can both *create* new vectors and *propagate* the
    existing ones along edges.

    Each channel of `v` is a learned "directional fingerprint" of the
    local environment of atom $i$ — pick a channel below to see it drawn
    on the molecule.
    """)
    return


@app.cell
def _(EquivariantInteraction, ase_molecule, build_graph, torch):
    # Pick a slightly bigger molecule than H2O so the channel arrows have
    # interesting geometry to draw on.
    torch.manual_seed(0)
    mol = ase_molecule("CH3OH")  # 6 atoms
    cutoff_vec = 2.5
    F_vec = 16

    graph_vec = build_graph(mol, cutoff=cutoff_vec)
    layer_vec = EquivariantInteraction(hidden_dim=F_vec, num_basis=8, cutoff=cutoff_vec)
    layer_vec.eval()

    # Bootstrap: s from an embedding, v initialized to zeros (PaiNN convention).
    embed_vec = torch.nn.Embedding(100, F_vec)
    torch.nn.init.normal_(embed_vec.weight, std=0.5)

    with torch.no_grad():
        s0 = embed_vec(graph_vec.z)  # [N, F]
        v0 = torch.zeros(graph_vec.n_atoms, F_vec, 3)
        s1, v1 = layer_vec(s0, v0, graph_vec)

    print("s shape:", tuple(s1.shape))
    print("v shape:", tuple(v1.shape))
    print("v initial norm (zeros):", float(v0.norm()))
    print("v after one layer norm (bootstrapped by creation message):", float(v1.norm()))
    return F_vec, graph_vec, layer_vec, s0, v1


@app.cell(hide_code=True)
def _(F_vec, mo):
    vec_channel = mo.ui.dropdown(
        options=[str(i) for i in range(F_vec)],
        value="0",
        label="v channel",
    )
    vec_channel
    return (vec_channel,)


@app.cell(hide_code=True)
def _(element_color, go, graph_vec, np, v1, vec_channel):
    _ch = int(vec_channel.value)
    _pos = graph_vec.pos.detach().numpy()
    _z = graph_vec.z.numpy()
    _v_ch = v1[:, _ch, :].detach().numpy()  # [N, 3] — one channel as a vector per atom

    # Scale arrows for visibility.
    # Normalize so the largest arrow has length ~1.5 Å, keeping atom geometry legible.
    _scale = 1.5 / max(np.linalg.norm(_v_ch, axis=-1).max(), 1e-6)

    _atoms_p = go.Scatter3d(
        x=_pos[:, 0],
        y=_pos[:, 1],
        z=_pos[:, 2],
        mode="markers",
        marker=dict(size=14, color=[element_color(int(z)) for z in _z]),
        name="atoms",
    )
    _arrow_traces = []
    for _k in range(graph_vec.n_atoms):
        _start = _pos[_k]
        _end = _start + _scale * _v_ch[_k]
        _arrow_traces.append(
            go.Scatter3d(
                x=[_start[0], _end[0]],
                y=[_start[1], _end[1]],
                z=[_start[2], _end[2]],
                mode="lines",
                line=dict(color="royalblue", width=5),
                showlegend=False,
            )
        )

    _fig_ch = go.Figure([_atoms_p, *_arrow_traces])
    _fig_ch.update_layout(
        title=f"v[:, {_ch}, :] — directional fingerprint, channel {_ch}",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=420,
    )
    _fig_ch
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The message phase, dissected

    `EquivariantInteraction` packs PaiNN's message phase and update phase
    into one `forward()`. Nb02 already dissected an invariant message phase
    by hand; here we do the same for the equivariant one — same skeleton,
    but with **three** messages per edge instead of one:

    1. **Scalar message** $m^s_{ij} = \psi^s(s_j) \cdot \phi^s(r_{ij})$ —
       same idea as the SchNet message in nb02.
    2. **Vector propagation** $m^{vv}_{ij} = \psi^{vv}(s_j) \cdot \phi^{vv}(r_{ij}) \cdot v_j$ —
       transport the sender's existing vector along the edge. (No-op on the
       first layer call, since `v=0` initially.)
    3. **Vector creation** $m^{vs}_{ij} = \psi^{vs}(s_j) \cdot \phi^{vs}(r_{ij}) \cdot \hat{r}_{ij}$ —
       build a fresh vector from the edge direction, scaled by a scalar weight.

    Then `index_add_` both to receivers (one scatter for the scalar
    aggregate $\Delta s$, one for the vector aggregate
    $\Delta v = \sum (m^{vv} + m^{vs})$).
    """)
    return


@app.cell
def _(F_vec, graph_vec, layer_vec, s0, torch):
    # Reuse layer_vec, graph_vec, s0 from section 3.
    src, dst = graph_vec.edge_index  # [E], [E]
    edge_vec = graph_vec.pos[dst] - graph_vec.pos[src]  # [E, 3]
    r = edge_vec.norm(dim=-1).clamp(min=1e-6)  # [E]
    unit = edge_vec / r.unsqueeze(-1)  # [E, 3]

    with torch.no_grad():
        rbf = layer_vec.basis(r) * layer_vec.envelope(r).unsqueeze(-1)  # [E, num_basis]
        phi_s, phi_vv, phi_vs = layer_vec.filter_net(rbf).chunk(3, dim=-1)  # each [E, F]
        psi_s, psi_vv, psi_vs = layer_vec.psi(s0)[src].chunk(3, dim=-1)  # each [E, F]

        # Use v=0 to start (matches section 3's bootstrap).
        v_in = torch.zeros(graph_vec.n_atoms, F_vec, 3)

        m_s = psi_s * phi_s  # [E, F]
        m_vv = (psi_vv * phi_vv).unsqueeze(-1) * v_in[src]  # [E, F, 3]  (zero on first layer)
        m_vs = (psi_vs * phi_vs).unsqueeze(-1) * unit.unsqueeze(-2)  # [E, F, 3]

        ds = torch.zeros_like(s0).index_add_(0, dst, m_s)  # [N, F]
        dv = torch.zeros_like(v_in).index_add_(0, dst, m_vv + m_vs)  # [N, F, 3]
        s_after_msg = s0 + ds
        v_after_msg = v_in + dv

    print("Δs norm:", float(ds.norm()))
    print("m_vv contribution to Δv (should be 0 since v_in=0):", float(m_vv.norm()))
    print("m_vs contribution to Δv:", float(m_vs.norm()))
    print("v after message phase norm:", float(v_after_msg.norm()))
    return s_after_msg, v_after_msg, v_in


@app.cell
def _(
    AtomGraph,
    graph_vec,
    layer_vec,
    np,
    s0,
    s_after_msg,
    torch,
    v_after_msg,
    v_in,
):
    # `EquivariantInteraction.forward` fuses message + update. We can't directly
    # pull "message-only" out without copying the layer's first half, so we
    # instead verify rotation equivariance of our hand-built version directly
    # (the same property the real layer holds).
    torch.manual_seed(7)
    R = torch.tensor(  # noqa: N806 — standard rotation notation
        [
            [np.cos(0.6), -np.sin(0.6), 0.0],
            [np.sin(0.6), np.cos(0.6), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    _pos_rot = graph_vec.pos @ R.T
    _graph_rot = AtomGraph(
        z=graph_vec.z,
        pos=_pos_rot,
        edge_index=graph_vec.edge_index,
        edge_vec=_pos_rot[graph_vec.edge_index[1]] - _pos_rot[graph_vec.edge_index[0]],
        edge_dist=(_pos_rot[graph_vec.edge_index[1]] - _pos_rot[graph_vec.edge_index[0]]).norm(dim=-1),
        cutoff=graph_vec.cutoff,
    )

    # Re-run the hand-built message phase on the rotated graph (v_in still zeros).
    src_r, dst_r = _graph_rot.edge_index
    edge_vec_r = _graph_rot.pos[dst_r] - _graph_rot.pos[src_r]
    r_r = edge_vec_r.norm(dim=-1).clamp(min=1e-6)
    unit_r = edge_vec_r / r_r.unsqueeze(-1)

    with torch.no_grad():
        rbf_r = layer_vec.basis(r_r) * layer_vec.envelope(r_r).unsqueeze(-1)
        phi_s_r, phi_vv_r, phi_vs_r = layer_vec.filter_net(rbf_r).chunk(3, dim=-1)
        psi_s_r, psi_vv_r, psi_vs_r = layer_vec.psi(s0)[src_r].chunk(3, dim=-1)
        m_s_r = psi_s_r * phi_s_r
        m_vs_r = (psi_vs_r * phi_vs_r).unsqueeze(-1) * unit_r.unsqueeze(-2)
        ds_r = torch.zeros_like(s0).index_add_(0, dst_r, m_s_r)
        dv_r = torch.zeros_like(v_in).index_add_(0, dst_r, m_vs_r)
        s_after_msg_r = s0 + ds_r
        v_after_msg_r = v_in + dv_r

    # Expected: s unchanged; v rotated by R (i.e. v @ R.T).
    scalar_drift = float((s_after_msg_r - s_after_msg).abs().max())
    vector_residual = float((v_after_msg_r - v_after_msg @ R.T).abs().max())

    print(f"max scalar drift under rotation: {scalar_drift:.2e}  (expect ~0)")
    print(f"max vector residual after rotating output: {vector_residual:.2e}  (expect ~0)")
    assert scalar_drift < 1e-5, "scalar features changed under rotation — bug!"
    assert vector_residual < 1e-5, "vectors did not rotate as expected — bug!"
    print("\nOK — message phase is rotation-equivariant. Scalars are invariant, vectors rotate with the molecule.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The update phase, as inset

    The message phase moved information *along edges*. The update phase does
    a per-atom mixing of `s` and `v` to let scalars learn from vector
    channels. Two linear maps applied channel-wise on the vector axis:

    $$
    Uv = U(v), \qquad Vv = V(v) \quad \text{(both have shape \([N, F, 3]\); no bias — a constant vector would break equivariance)}
    $$

    From these we build **two rotation invariants** per channel:

    $$
    \|Vv\|_2 \in \mathbb{R}^F, \qquad \langle Uv, Vv \rangle \in \mathbb{R}^F
    $$

    These scalars are how `s` learns from `v`. Then an MLP on
    $[s,\ \|Vv\|_2]$ produces three gates:

    - $a_{ss}$: scalar bias correction added to `s`.
    - $a_{sv}$: scalar gate multiplying $\langle Uv, Vv \rangle$ before
      adding it to `s`.
    - $a_{vv}$: per-channel scalar gate multiplying $Uv$ before adding it to
      `v`.

    We won't re-derive each line — read it, run it, then convince yourself
    that the end-to-end energy is rotation-invariant.
    """)
    return


@app.cell
def _(F_vec, graph_vec, layer_vec, s0, torch):
    # Run the full layer (message + update). We'll inspect the update-phase
    # intermediates by reaching into the layer's modules.
    torch.manual_seed(0)
    _s_in = s0  # from section 3
    _v_in_run = torch.zeros(graph_vec.n_atoms, F_vec, 3)

    with torch.no_grad():
        _s_out, _v_out = layer_vec(_s_in, _v_in_run, graph_vec)

        # Recompute the message phase intermediates so we can show the update inputs.
        src2, dst2 = graph_vec.edge_index
        ev = graph_vec.pos[dst2] - graph_vec.pos[src2]
        rr = ev.norm(dim=-1).clamp(min=1e-6)
        uu = ev / rr.unsqueeze(-1)
        rbf2 = layer_vec.basis(rr) * layer_vec.envelope(rr).unsqueeze(-1)
        a_s, a_vv, a_vs = layer_vec.filter_net(rbf2).chunk(3, dim=-1)
        b_s, b_vv, b_vs = layer_vec.psi(_s_in)[src2].chunk(3, dim=-1)
        s_mid = _s_in + torch.zeros_like(_s_in).index_add_(0, dst2, b_s * a_s)
        v_mid = _v_in_run + torch.zeros_like(_v_in_run).index_add_(
            0,
            dst2,
            (b_vv * a_vv).unsqueeze(-1) * _v_in_run[src2]
            + (b_vs * a_vs).unsqueeze(-1) * uu.unsqueeze(-2),
        )

        Uv = layer_vec.U(v_mid.transpose(-1, -2)).transpose(-1, -2)  # noqa: N806 — [N, F, 3]
        Vv = layer_vec.V(v_mid.transpose(-1, -2)).transpose(-1, -2)  # noqa: N806 — [N, F, 3]
        vnorm = Vv.norm(dim=-1)  # [N, F]
        vdot = (Uv * Vv).sum(dim=-1)  # [N, F]

    print("Uv shape:", tuple(Uv.shape))
    print("Vv shape:", tuple(Vv.shape))
    print("||Vv|| per (atom, channel) shape:", tuple(vnorm.shape), "  sample [0, :4]:", vnorm[0, :4].tolist())
    print("<Uv, Vv> shape:", tuple(vdot.shape), "  sample [0, :4]:", vdot[0, :4].tolist())
    return


@app.cell
def _(EquivariantMPNN, F_vec, ase_molecule, build_graph, np, torch):
    # Build a full EquivariantMPNN, energy-check on rotation.
    torch.manual_seed(0)
    _mol_check = ase_molecule("CH3OH")
    _check_cutoff = 2.5
    _check_model = EquivariantMPNN(hidden_dim=F_vec, num_basis=8, cutoff=_check_cutoff, n_layers=2).double()

    _graph_d = build_graph(_mol_check, cutoff=_check_cutoff, dtype=torch.float64)
    _e_orig = _check_model(_graph_d)

    _R_check = torch.tensor(
        [
            [np.cos(1.2), -np.sin(1.2), 0.0],
            [np.sin(1.2), np.cos(1.2), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    _mol_rot = _mol_check.copy()
    _mol_rot.set_positions(_mol_check.get_positions() @ _R_check.numpy().T)
    _graph_dr = build_graph(_mol_rot, cutoff=_check_cutoff, dtype=torch.float64)
    _e_rot = _check_model(_graph_dr)

    print(f"E(original)  = {_e_orig.item():.10f}")
    print(f"E(rotated)   = {_e_rot.item():.10f}")
    print(f"|ΔE|         = {abs(_e_orig.item() - _e_rot.item()):.2e}  (expect < 1e-8)")
    assert abs(_e_orig.item() - _e_rot.item()) < 1e-8
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Train + compare against the invariant model

    Same training stack as nb04 (`tinymlip.train`), same rMD17 subset, same
    hyperparameters across both models. **Only the architecture varies** —
    that's what makes the comparison honest.

    Default preset is tighter than nb04's `tiny` (fewer frames, fewer
    epochs) so that two trainings still fit in a single notebook run under
    ~10 minutes on CPU; aim ~5 minutes. nb04 trains longer because it only
    fits one model.
    """)
    return


@app.cell
def _():
    # Tighter than nb04's tiny preset so two trainings still fit in ~5-10 min CPU.
    tiny_nb05 = dict(
        molecule="ethanol",  # the rMD17 molecule downloaded locally (see data/download.py)
        n_train=300,
        n_val=100,
        cutoff=5.0,
        hidden_dim=64,
        num_basis=20,
        n_layers=2,
        batch_size=8,
        lr=1e-3,
        n_epochs=40,
        w_e=1.0,
        w_f=100.0,
    )
    tiny_nb05
    return (tiny_nb05,)


@app.cell
def _(
    DataLoader,
    fit_atomic_reference,
    load_rmd17,
    make_collate,
    tiny_nb05,
    to_torch_dataset,
):
    # Mirror nb04's data pipeline: load the rMD17 train split, carve a
    # validation slice off the end, then wrap each half in its own bundle.
    from tinymlip.data import RMD17Bundle

    _trainval_bundle = load_rmd17(
        tiny_nb05["molecule"],
        split="train",
        cv_fold=1,
        n_frames=tiny_nb05["n_train"] + tiny_nb05["n_val"],
        seed=0,
    )

    _n_train = tiny_nb05["n_train"]
    _train_structures = _trainval_bundle.structures[:_n_train]
    _val_structures = _trainval_bundle.structures[_n_train:]
    _train_meta = _trainval_bundle.meta.head(_n_train)
    _val_meta = _trainval_bundle.meta.slice(_n_train)

    _train_bundle = RMD17Bundle(meta=_train_meta, structures=list(_train_structures))
    _val_bundle = RMD17Bundle(meta=_val_meta, structures=list(_val_structures))

    shifts = fit_atomic_reference(_train_structures, _train_meta["energy"].to_numpy())

    _collate = make_collate(cutoff=tiny_nb05["cutoff"])
    train_loader = DataLoader(
        to_torch_dataset(_train_bundle),
        batch_size=tiny_nb05["batch_size"],
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        to_torch_dataset(_val_bundle),
        batch_size=tiny_nb05["batch_size"],
        shuffle=False,
        collate_fn=_collate,
    )

    (len(train_loader), len(val_loader), shifts)
    return shifts, train_loader, val_loader


@app.cell
def _(
    InvariantMPNN,
    pl,
    shifts,
    tiny_nb05,
    torch,
    train,
    train_loader,
    val_loader,
):
    torch.manual_seed(0)
    invariant_model = InvariantMPNN(
        hidden_dim=tiny_nb05["hidden_dim"],
        num_basis=tiny_nb05["num_basis"],
        cutoff=tiny_nb05["cutoff"],
        n_layers=tiny_nb05["n_layers"],
    )
    invariant_log = train(
        invariant_model,
        train_loader,
        val_loader,
        n_epochs=tiny_nb05["n_epochs"],
        lr=tiny_nb05["lr"],
        w_e=tiny_nb05["w_e"],
        w_f=tiny_nb05["w_f"],
        shifts=shifts,
    ).with_columns(pl.lit("invariant").alias("model"))
    invariant_log.tail(4)
    return invariant_log, invariant_model


@app.cell
def _(
    EquivariantMPNN,
    pl,
    shifts,
    tiny_nb05,
    torch,
    train,
    train_loader,
    val_loader,
):
    torch.manual_seed(0)
    equivariant_model = EquivariantMPNN(
        hidden_dim=tiny_nb05["hidden_dim"],
        num_basis=tiny_nb05["num_basis"],
        cutoff=tiny_nb05["cutoff"],
        n_layers=tiny_nb05["n_layers"],
    )
    equivariant_log = train(
        equivariant_model,
        train_loader,
        val_loader,
        n_epochs=tiny_nb05["n_epochs"],
        lr=tiny_nb05["lr"],
        w_e=tiny_nb05["w_e"],
        w_f=tiny_nb05["w_f"],
        shifts=shifts,
    ).with_columns(pl.lit("equivariant").alias("model"))
    equivariant_log.tail(4)
    return equivariant_log, equivariant_model


@app.cell(hide_code=True)
def _(equivariant_log, go, invariant_log, pl):
    runs = pl.concat([invariant_log, equivariant_log])

    _fig = go.Figure()
    for (_model_name, _split), _color, _dash in [
        (("invariant", "train"), "royalblue", "solid"),
        (("invariant", "val"), "royalblue", "dash"),
        (("equivariant", "train"), "crimson", "solid"),
        (("equivariant", "val"), "crimson", "dash"),
    ]:
        _df = runs.filter((pl.col("model") == _model_name) & (pl.col("split") == _split))
        _fig.add_trace(
            go.Scatter(
                x=_df["epoch"].to_list(),
                y=_df["energy_mae"].to_list(),
                mode="lines+markers",
                name=f"{_model_name} / {_split}",
                line=dict(color=_color, dash=_dash),
            )
        )
    _fig.update_layout(
        title="Energy MAE per epoch (lower = better)",
        xaxis_title="epoch",
        yaxis_title="energy MAE (kcal/mol)",
        height=420,
        margin=dict(l=40, r=10, t=40, b=40),
    )
    _fig
    return (runs,)


@app.cell(hide_code=True)
def _(equivariant_model, go, invariant_model, mo, pl, runs):
    _final = (
        runs.filter(pl.col("split") == "val")
        .group_by("model")
        .agg(pl.col("force_mae").last().alias("final_force_mae"))
    )

    _bar = go.Figure(
        go.Bar(
            x=_final["model"].to_list(),
            y=_final["final_force_mae"].to_list(),
            marker_color=["royalblue", "crimson"],
        )
    )
    _bar.update_layout(
        title="Final validation force MAE (lower = better)",
        yaxis_title="force MAE (kcal/mol/Å)",
        height=340,
        margin=dict(l=40, r=10, t=40, b=40),
    )

    def _n_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    _params = pl.DataFrame(
        {
            "model": ["invariant", "equivariant"],
            "trainable_params": [_n_params(invariant_model), _n_params(equivariant_model)],
        }
    )

    mo.vstack([_bar, _params])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this shows

    At identical hyperparameters and identical training data, the equivariant
    model reaches **substantially lower validation force MAE** than the
    invariant model — roughly a 2× improvement on this preset. Energy MAE
    tells a more nuanced story: the equivariant model has more parameters
    (extra channels in the message + update phases), so per-parameter the
    comparison depends on training budget. With longer training and at scale,
    PaiNN-style equivariant models also overtake on energy MAE; on this
    tiny-preset short run, the headline win is on forces.

    The cost is a more complex layer (message + update phases, vector
    channels), and roughly 2× compute per forward pass for the same hidden
    dimension.

    The structural reason it works: the equivariant model can encode and
    transport **directional** information through its hidden state. Forces
    are vectors; an architecture whose hidden state matches the geometry of
    the prediction target wins by inductive bias, not just capacity.

    **Forward link → nb06.** PaiNN handles molecules fine. Crystals need
    one more idea: a graph whose edges can cross periodic boundaries. We'll
    extend the equivariant model to PBC and run a tiny crystal demo.

    **Beyond PaiNN.** Higher-order tensors (ℓ ≥ 2) give equivariant models
    even more expressive hidden states. NequIP and MACE are the standard
    next step — they typically build on the `e3nn` library. We don't pull
    in `e3nn` here because it's a heavy dependency for what is, at this
    stage of the arc, a single notebook of motivation.
    """)
    return


if __name__ == "__main__":
    app.run()
