import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 · Message passing

    **What this notebook teaches.** A message-passing step on an atom graph
    is just three operations: every atom *gathers* features from its
    neighbors, *aggregates* them with a sum, and *updates* itself from the
    aggregate. The same shape recurs in every modern MLIP. We build it from
    scratch (five lines), then promote it to a SchNet-style
    `InvariantInteraction` layer with a learnable radial filter.

    **Prerequisites.** Notebook 01 (atoms as graphs), basic `torch.nn`.

    **By the end you can:**
    - Read a message-passing forward pass and name each of the three steps.
    - Explain in one sentence what a radial basis does and why the cutoff
      envelope multiplies it.
    - Run one `InvariantInteraction` step on ethanol and read off how each
      atom's features changed.
    """)
    return


@app.cell
def _():
    from tinymlip import load_rmd17

    bundle = load_rmd17("ethanol", split="train", n_frames=1, seed=0)
    atoms = bundle.structures[0]
    atoms
    return (atoms,)


@app.cell(hide_code=True)
def _(mo):
    cutoff = mo.ui.slider(start=2.0, stop=6.0, step=0.1, value=5.0, label="cutoff (Å)")
    cutoff
    return (cutoff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three steps: gather, aggregate, update

    For each atom *i*:

    1. **Gather:** look at every neighbor *j* connected by an edge.
    2. **Aggregate:** combine those neighbor features into one vector. We use
       a **sum** rather than a mean for one specific reason: *size
       extensivity*. If you put two copies of a system far apart, the energy
       must double — and that only happens when per-atom features (and the
       energy built from them) scale linearly with system size. A mean would
       erase that scaling. (We unpack this in nb03 — Gilmer et al. 2017
       call this gather/aggregate/update decomposition the "MPNN framework.")
    3. **Update:** mix the aggregate back into atom *i*'s own feature.

    The next cell does exactly this on ethanol with no learnable weights at
    all — just to see the shape of message passing before we add machinery.
    """)
    return


@app.cell
def _(atoms, cutoff, mo):
    import torch

    from tinymlip import build_graph

    graph = build_graph(atoms, cutoff=cutoff.value)
    torch.manual_seed(0)

    # 4 random scalar features per atom — pretend these are learned embeddings.
    atom_features = torch.randn(graph.n_atoms, 4)  # [n_atoms, n_features]

    # 1) Gather: every edge picks up the sender's feature.
    senders, receivers = graph.edge_index  # each: [n_edges]
    messages = atom_features[senders]  # [n_edges, n_features]

    # 2) Aggregate: each receiver sums its incoming messages.
    # index_add_(0, receivers, messages): for each edge e, add messages[e]
    # into row receivers[e] of the destination — a scatter-sum.
    aggregated_messages = torch.zeros_like(atom_features).index_add_(
        0,
        receivers,
        messages,
    )  # [n_atoms, n_features]

    # 3) Update: residual sum of own feature + neighborhood.
    atom_features_next = atom_features + aggregated_messages  # [n_atoms, n_features]

    # Per-atom L2 change tells you which atoms moved the most.
    per_atom_change = (atom_features_next - atom_features).norm(dim=-1)  # [n_atoms]
    mo.md(
        f"`graph.n_atoms` = {graph.n_atoms}, `graph.n_edges` = {graph.n_edges}.\n\n"
        "Per-atom feature change after one naive MPNN step:\n\n"
        + "\n".join(
            f"- atom {i} (Z={int(graph.z[i])}): change = {per_atom_change[i].item():.3f}"
            for i in range(graph.n_atoms)
        )
    )
    return build_graph, graph, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What's missing

    Three problems with the cell above:

    1. **Distance is ignored.** A neighbor 1 Å away contributes exactly as
       much as one 4 Å away. That's not physics — atoms feel each other
       through smooth, distance-dependent interactions.
    2. **No learnable transform.** The update was just a residual sum;
       nothing for the model to fit to data.
    3. **Chemistry is missing.** We used random per-atom features above. In
       a real model (nb04), an *embedding table* maps each atomic number
       `z` to an initial `hidden_dim` vector — every carbon atom starts
       with one feature vector, every hydrogen with another. We kept the
       features random here to keep the focus on the message-passing
       mechanics; nb04 is where chemistry enters the features.

    **The fix for problems 1 and 2: give each edge a *learnable* weight
    that depends on its distance.** We do this in two steps:

    1. **Expand the distance `r` into a vector of features.** Think of
       `num_basis = 8` as eight smooth "distance detectors" — one fires
       loudest near 1 Å, another near 2 Å, and so on. An edge at 1 Å
       lights up the short-range detectors strongly and the others
       weakly; an edge at 4 Å does the opposite. A single number `r`
       becomes a richer fingerprint the model can work with.
    2. **Mix those detector readings into a weight.** A small learnable
       layer combines the 8 detector values into a per-edge weight that
       scales the message. After training, this layer can say "amplify
       messages from neighbors at 1 Å, suppress messages from neighbors
       at 4 Å" — whatever the data calls for.

    *Why bother expanding `r`?* A linear layer applied to a single number
    can only learn a straight line in *r* — to fit a wiggly
    distance-dependence the model would otherwise need many deep nonlinear
    layers to relearn what "distance" means from scratch. With a
    vocabulary of localized distance features, the same wiggly target
    becomes a cheap linear combination of bumps — the same trick
    transformers use with positional embeddings and kernel methods use
    with RBFs. The cosine envelope multiplies in on top to drive every
    basis feature smoothly to zero at the cutoff, so atoms drifting
    across the cutoff boundary don't cause sudden jumps in energy or
    forces.

    Together this is called a *continuous-filter convolution* (SchNet,
    Schütt et al. 2018) — "continuous" because the weight changes
    smoothly with distance, not in bins. The first piece (the eight
    detectors) is what we look at next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Same three steps, with knobs

    The `InvariantInteraction` layer below is still gather → aggregate →
    update — just with three small networks slotted in:

    - `phi_s` runs **before** gather: it transforms each sender's features
      before they ride out as messages.
    - `W(r_ij)` runs **inside** aggregate: a tiny MLP (`filter_net`)
      reads the edge's distance fingerprint and produces a per-edge,
      per-channel weight that scales the message.
    - `phi_u` runs **after** aggregate: it mixes the summed neighborhood
      back into the receiver's own features.

    Everything we're about to plot is the shape of `W(r_ij)` — the only
    piece that depends on distance. The fingerprint bar chart below is
    exactly what `filter_net` reads per edge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    num_basis = mo.ui.slider(start=4, stop=20, step=1, value=8, label="num_basis")
    num_basis
    return (num_basis,)


@app.cell(hide_code=True)
def _(cutoff, num_basis, r_demo, torch):
    import plotly.graph_objects as go

    from tinymlip import BesselBasis, CosineEnvelope

    basis = BesselBasis(num_basis=num_basis.value, cutoff=cutoff.value)
    env = CosineEnvelope(cutoff=cutoff.value)

    # Shared palette: each basis function gets one consistent color across the
    # curves plot, the fingerprint bars below, and the per-channel comparison.
    # 20 entries covers num_basis up to 20.
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d3",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]

    r_grid = torch.linspace(0.05, cutoff.value, 200)
    b = basis(r_grid).detach().numpy()
    e = env(r_grid).detach().numpy()

    fig_basis = go.Figure()
    for i_basis in range(num_basis.value):
        fig_basis.add_trace(
            go.Scatter(
                x=r_grid.numpy(),
                y=b[:, i_basis],
                mode="lines",
                name=f"b_{i_basis + 1}",
                showlegend=False,
                line=dict(color=palette[i_basis % len(palette)], width=1.4),
            )
        )
    fig_basis.add_trace(
        go.Scatter(
            x=r_grid.numpy(),
            y=e * b.max(),
            mode="lines",
            name="envelope (scaled)",
            line=dict(dash="dash", color="black", width=2),
        )
    )
    # Vertical "scanner" line at r_demo — the bar chart below is the slice of
    # the curves at this x position.
    fig_basis.add_vline(
        x=r_demo.value,
        line=dict(dash="dot", color="crimson", width=2),
        annotation_text=f"r = {r_demo.value:.2f} Å",
        annotation_position="top",
    )
    fig_basis.update_layout(
        title=f"Bessel basis ({num_basis.value} functions) × cosine envelope, cutoff={cutoff.value} Å",
        xaxis_title="r (Å)",
        yaxis_title="b_n(r)",
        height=380,
    )
    fig_basis
    return basis, env, go, palette


@app.cell(hide_code=True)
def _(cutoff, mo):
    r_demo = mo.ui.slider(
        start=0.3,
        stop=cutoff.value,
        step=0.05,
        value=min(1.5, cutoff.value),
        label="r to decompose (Å)",
    )
    r_demo
    return (r_demo,)


@app.cell(hide_code=True)
def _(basis, env, go, num_basis, palette, r_demo, torch):
    r_demo_t = torch.tensor([r_demo.value])
    decomposition = (basis(r_demo_t) * env(r_demo_t).unsqueeze(-1)).detach().squeeze(0).numpy()

    fig_decomp = go.Figure(
        data=[
            go.Bar(
                x=[f"b_{i + 1}" for i in range(num_basis.value)],
                y=decomposition,
                marker=dict(color=[palette[i % len(palette)] for i in range(num_basis.value)]),
            )
        ]
    )
    fig_decomp.update_layout(
        title=f"Fingerprint of r = {r_demo.value:.2f} Å (basis × envelope)",
        xaxis_title="basis function",
        yaxis_title="b_n(r) · f_cut(r)",
        height=300,
    )
    fig_decomp
    return


@app.cell(hide_code=True)
def _(cutoff, go, graph, num_basis, palette, torch):
    from tinymlip import InvariantInteraction

    torch.manual_seed(0)
    layer = InvariantInteraction(
        hidden_dim=4,
        num_basis=num_basis.value,
        cutoff=cutoff.value,
    )

    with torch.no_grad():
        # Smooth curves: filter evaluated on a dense r grid covering the
        # whole [0, cutoff] range. This is the layer's "shape."
        r_dense = torch.linspace(0.3, cutoff.value, 200)
        rbf_dense = layer.basis(r_dense) * layer.envelope(r_dense).unsqueeze(-1)
        weights = layer.filter_net(rbf_dense)  # [200, hidden_dim]
        channel_mean = weights.mean(dim=0, keepdim=True)
        weights_rel = weights / channel_mean

        # Markers: filter evaluated at ethanol's actual edge distances,
        # normalized by the same channel means so dots land on the curves.
        edge_vec_e = graph.pos[graph.edge_index[1]] - graph.pos[graph.edge_index[0]]
        r_edges = edge_vec_e.norm(dim=-1).clamp(min=1e-6)
        rbf_edges = layer.basis(r_edges) * layer.envelope(r_edges).unsqueeze(-1)
        weights_edges = layer.filter_net(rbf_edges)  # [E, hidden_dim]
        weights_edges_rel = weights_edges / channel_mean

    fig_compare = go.Figure()
    fig_compare.add_hline(
        y=1.0,
        line=dict(dash="dash", color="grey"),
        annotation_text="naive MPNN: uniform contribution (every edge = channel mean)",
        annotation_position="top left",
    )
    for ch in range(weights_rel.shape[1]):
        color = palette[ch % len(palette)]
        # Smooth curve over r — the layer's shape.
        fig_compare.add_trace(
            go.Scatter(
                x=r_dense.numpy(),
                y=weights_rel[:, ch].numpy(),
                mode="lines",
                line=dict(color=color, width=2),
                name=f"channel {ch}",
                legendgroup=f"ch{ch}",
            )
        )
        # Dots at actual edge distances — where this molecule's data lives.
        fig_compare.add_trace(
            go.Scatter(
                x=r_edges.detach().numpy(),
                y=weights_edges_rel[:, ch].detach().numpy(),
                mode="markers",
                marker=dict(size=4, color=color, line=dict(color="white", width=0.5)),
                showlegend=False,
                legendgroup=f"ch{ch}",
                hovertemplate=f"channel {ch}<br>r = %{{x:.2f}} Å<br>rel weight = %{{y:.3f}}<extra></extra>",
            )
        )
    fig_compare.update_layout(
        title=f"InvariantInteraction filter output across r (untrained, normalized per channel; dots = ethanol's {graph.n_edges} edges)",
        xaxis_title="r (Å)",
        yaxis_title="weight relative to channel mean",
        height=420,
    )
    fig_compare
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading this plot

    Each colored curve is one of the hidden channels (we picked
    `hidden_dim = 4`). Weights are normalized per channel, so **y = 1
    is the naive MPNN baseline** — a channel that treats every edge the
    same lives exactly on that line. Swings above 1 = "amplify messages
    from this distance"; below 1 = "dampen them."

    - **Channels can specialize.** Some curves swing well above and
      below 1 — that channel learned to weigh nearby and far-away
      neighbors differently. Others sit close to 1 — that channel
      chose to ignore distance for this random init.
    - **With random init the swings are subtle.** That's expected:
      we haven't trained anything yet. The point isn't the magnitude
      of the swings — it's that the *capacity* to swing is built in.
      A trained model will push these curves into sharp, meaningful
      shapes (one channel might lock onto C–H bond lengths around 1 Å,
      another onto O–H distances, and so on).
    - **The curves are smooth.** That's geometry, not learning: the
      Bessel × envelope basis is smooth in *r*, so any linear
      combination of it is too. That smoothness is what keeps forces
      well-defined when atoms cross the cutoff boundary.

    Move the cutoff or `num_basis` sliders and watch the shapes shift —
    you're seeing different *capacity profiles*, not different trained
    behavior.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stacking layers grows the receptive field

    A single message-passing step lets atom *i* see only its direct
    neighbors. Stack *k* layers and atom *i* can see anything reachable
    in *k* hops on the graph. **This is pure graph topology — it
    doesn't depend on the model's weights at all.** With training, the
    model decides what to *do* with that receptive field; without
    training, the structural reach is still set by depth.

    To see this clearly we drop to a tighter cutoff for the next plot —
    at the cutoff slider's default (≈ 5 Å), ethanol's graph is fully
    connected and depth 1 already covers everything. With a 1.6 Å
    cutoff we get the covalent-bond graph (C–H, C–C, C–O, O–H) and the
    receptive field grows visibly with depth.
    """)
    return


@app.cell(hide_code=True)
def _(atom_labels, go, graph_sparse):
    from tinymlip.viz import element_color, element_radius

    # Lines for each covalent bond (the sparse graph's edges).
    bond_x, bond_y, bond_z = [], [], []
    src_s, dst_s = graph_sparse.edge_index
    for s_i, d_i in zip(src_s.tolist(), dst_s.tolist(), strict=True):
        if s_i < d_i:  # one direction only
            bond_x += [float(graph_sparse.pos[s_i, 0]), float(graph_sparse.pos[d_i, 0]), None]
            bond_y += [float(graph_sparse.pos[s_i, 1]), float(graph_sparse.pos[d_i, 1]), None]
            bond_z += [float(graph_sparse.pos[s_i, 2]), float(graph_sparse.pos[d_i, 2]), None]

    pos_3d = graph_sparse.pos.numpy()
    z_3d = graph_sparse.z.numpy()

    fig_3d = go.Figure()
    fig_3d.add_trace(
        go.Scatter3d(
            x=bond_x,
            y=bond_y,
            z=bond_z,
            mode="lines",
            line=dict(color="#888", width=4),
            showlegend=False,
            hoverinfo="skip",
            name="bonds",
        )
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=pos_3d[:, 0],
            y=pos_3d[:, 1],
            z=pos_3d[:, 2],
            mode="markers+text",
            marker=dict(
                size=[element_radius(int(z)) * 14 for z in z_3d],
                color=[element_color(int(z)) for z in z_3d],
                line=dict(color="#222", width=1),
            ),
            text=atom_labels,
            textposition="top center",
            textfont=dict(size=11, color="#111"),
            hoverinfo="skip",
            showlegend=False,
            name="atoms",
        )
    )
    fig_3d.update_layout(
        title="Ethanol with atom indices (matches the heatmap below)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            dragmode="turntable",
        ),
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig_3d
    return


@app.cell(hide_code=True)
def _(atoms, build_graph, go, torch):
    import numpy as np
    from ase.data import chemical_symbols

    # Pure-topology demo: build a tighter graph on the same molecule so the
    # "depth = hops" lesson is visible. At the notebook\'s cutoff=5 Å,
    # ethanol is fully connected (every atom is everyone\'s neighbor at
    # depth 1) and stacking layers cannot grow the receptive field. With
    # demo_cutoff=1.6 Å we get ethanol\'s covalent-bond graph: short, sparse,
    # but still connected through the central C-C bond.
    demo_cutoff = 1.6
    graph_sparse = build_graph(atoms, cutoff=demo_cutoff)

    n = graph_sparse.n_atoms
    adjacency = torch.zeros(n, n)
    adjacency[graph_sparse.edge_index[0], graph_sparse.edge_index[1]] = 1.0
    # Add self-loops: an atom is trivially "reachable from itself" at depth 0.
    adjacency_self = adjacency + torch.eye(n)

    # Reachability at depth k = atoms reachable in <= k hops on the graph.
    # (Boolean matrix power: (A + I)^k > 0.)
    reach_rows = []
    power = adjacency_self.clone()
    for _depth in range(1, 4):
        reach_rows.append((power[0] > 0).float().numpy())  # row 0 = "from atom 0"
        power = ((power @ adjacency_self) > 0).float()

    heat = np.stack(reach_rows, axis=0)  # [3, n_atoms]

    # Element-and-index labels for the heatmap x-axis, e.g. "C[0]", "H[3]".
    atom_labels = [f"{chemical_symbols[int(graph_sparse.z[i])]}[{i}]" for i in range(n)]

    fig_field = go.Figure(
        data=go.Heatmap(
            z=heat,
            x=atom_labels,
            y=[f"depth {d}" for d in range(1, 4)],
            colorscale=[[0.0, "#f4f4f8"], [1.0, "#2ca02c"]],
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )
    fig_field.update_layout(
        title=f"Receptive field of atom 0 (ethanol, demo cutoff = {demo_cutoff} Å, {graph_sparse.n_edges} edges)",
        xaxis_title="input atom",
        yaxis_title="depth k",
        height=320,
    )
    fig_field
    return atom_labels, graph_sparse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we deferred (and where it goes)

    Two things are missing from this notebook on purpose:

    - **From features to energy.** A real MLIP collapses the per-atom
      features into a single number (the energy) and then takes its
      gradient w.r.t. positions to get forces. That's notebook 03.
    - **Direction-aware messages.** Our scalar messages discard the
      *direction* an edge points — only the distance enters through the
      radial basis. For some physics that's fine; for others it isn't.
      Notebook 05 introduces *equivariant* message passing, where each
      atom also carries a small set of vector channels that rotate with
      the molecule.
    """)
    return


if __name__ == "__main__":
    app.run()
