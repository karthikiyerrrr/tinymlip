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


@app.cell
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
    2. **Aggregate:** combine those neighbor features into one vector (we use
       a sum — order-independent, so the result doesn't depend on how the
       neighbors are listed).
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
    x = torch.randn(graph.n_atoms, 4)

    # 1) Gather: every edge picks up the sender's feature.
    src, dst = graph.edge_index
    m = x[src]  # [E, 4]

    # 2) Aggregate: receivers sum their incoming messages.
    agg = torch.zeros_like(x).index_add_(0, dst, m)  # [N, 4]

    # 3) Update: residual sum of own feature + neighborhood.
    x_new = x + agg

    # Per-atom L2 change tells you which atoms moved the most.
    change = (x_new - x).norm(dim=-1)
    mo.md(
        f"`graph.n_atoms` = {graph.n_atoms}, `graph.n_edges` = {graph.n_edges}.\n\n"
        "Per-atom feature change after one naive MPNN step:\n\n"
        + "\n".join(
            f"- atom {i} (Z={int(graph.z[i])}): change = {change[i].item():.3f}"
            for i in range(graph.n_atoms)
        )
    )

    return change, graph, torch, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What’s missing

    Two problems with the cell above:

    1. **Distance is ignored.** A neighbor 1 Å away contributes exactly as
       much as one 4 Å away. That’s not physics — atoms feel each other
       through smooth, distance-dependent interactions.
    2. **No learnable transform.** The "update" was just a residual sum;
       nothing for the model to fit to data.

    The fix: turn each edge distance *r* into a small vector of basis
    features, then learn a *filter* that maps those features to a per-edge
    weight on the message. That’s a *continuous-filter convolution*
    (SchNet, Schütt et al. 2018) — and the basis is what we look at next.
    """)
    return


@app.cell
def _(mo):
    num_basis = mo.ui.slider(start=4, stop=20, step=1, value=8, label="num_basis")
    num_basis

    return (num_basis,)


@app.cell
def _(cutoff, num_basis, torch):
    import plotly.graph_objects as go

    from tinymlip import BesselBasis, CosineEnvelope

    basis = BesselBasis(num_basis=num_basis.value, cutoff=cutoff.value)
    env = CosineEnvelope(cutoff=cutoff.value)

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
                line=dict(width=1.2),
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
    fig_basis.update_layout(
        title=f"Bessel basis ({num_basis.value} functions) × cosine envelope, cutoff={cutoff.value} Å",
        xaxis_title="r (Å)",
        yaxis_title="b_n(r)",
        height=380,
    )
    fig_basis

    return (go,)


@app.cell
def _(change, cutoff, go, graph, num_basis, torch, x):
    from tinymlip import InvariantInteraction

    torch.manual_seed(0)
    layer = InvariantInteraction(
        hidden_dim=4,
        num_basis=num_basis.value,
        cutoff=cutoff.value,
    )
    with torch.no_grad():
        x_layer = layer(x, graph)
    change_layer = (x_layer - x).norm(dim=-1)

    # Side-by-side comparison: naive (uniform) vs InvariantInteraction (radial filter).
    fig_compare = go.Figure()
    atom_idx = list(range(graph.n_atoms))
    fig_compare.add_trace(go.Bar(x=atom_idx, y=change.detach().numpy(), name="naive MPNN"))
    fig_compare.add_trace(
        go.Bar(x=atom_idx, y=change_layer.detach().numpy(), name="InvariantInteraction")
    )
    fig_compare.update_layout(
        title="Per-atom feature change after one step (ethanol)",
        xaxis_title="atom index",
        yaxis_title="‖Δx_i‖",
        barmode="group",
        height=360,
    )
    fig_compare

    return (InvariantInteraction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stacking layers grows the receptive field

    A single message-passing step lets atom *i* see only its direct
    neighbors. Stack *k* layers and atom *i* can see anything reachable in
    *k* hops on the graph. The next cell shows this directly: for each
    depth *k* ∈ {1, 2, 3}, we compute the gradient of atom 0's output (its
    summed features) with respect to every input atom. Bright cells mean
    "atom *j* influenced atom 0 at depth *k*."
    """)
    return


@app.cell
def _(InvariantInteraction, cutoff, go, graph, num_basis, torch, x):
    import numpy as np

    torch.manual_seed(0)
    layers_stack = [
        InvariantInteraction(hidden_dim=4, num_basis=num_basis.value, cutoff=cutoff.value)
        for _ in range(3)
    ]

    influence = []
    for depth in range(1, 4):
        x_req = x.detach().clone().requires_grad_(True)
        h = x_req
        for layer_d in layers_stack[:depth]:
            h = layer_d(h, graph)
        readout = h[0].sum()  # any scalar function of atom 0's features
        (g,) = torch.autograd.grad(readout, x_req)  # [n_atoms, 4]
        influence.append(g.norm(dim=-1).detach().numpy())

    heat = np.stack(influence, axis=0)  # [3, n_atoms]

    fig_field = go.Figure(
        data=go.Heatmap(
            z=heat,
            x=list(range(graph.n_atoms)),
            y=[f"depth {d}" for d in range(1, 4)],
            colorscale="Viridis",
            colorbar=dict(title="‖∂h₀ / ∂xⱼ‖"),
        )
    )
    fig_field.update_layout(
        title="Receptive field of atom 0 after k layers",
        xaxis_title="input atom j",
        yaxis_title="depth k",
        height=320,
    )
    fig_field

    return


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
