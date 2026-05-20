import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 · Energy and forces

    **What this notebook teaches.** A real MLIP turns the per-atom features
    from notebook 02 into a single number — the molecular energy — and then
    recovers forces by differentiating that energy with respect to the atomic
    positions. Three ideas, in order:

    1. **Atomic-number embedding.** Each atom becomes a learnable feature
       vector keyed by its element (H, C, O, …). That's the model's input.
    2. **Readout + sum.** A small per-atom MLP turns those features into a
       per-atom scalar; the molecular energy is the *sum* over atoms. The sum
       is what makes the energy *size-extensive* — twice as much molecule,
       twice as much energy.
    3. **Forces via autograd.** Forces are the negative gradient of the
       energy with respect to positions: $F_i = -\partial E / \partial \mathbf{r}_i$.
       `torch.autograd.grad` does the bookkeeping. No separate force head —
       ever. This guarantees the forces are *conservative*.

    **Prerequisites.** Notebook 02 (message passing).

    **By the end you can:**
    - Read an MLIP forward pass end-to-end and name the embedding, the
      interactions, the readout, and the sum.
    - Explain in one sentence why MLIP energies are size-extensive.
    - Compute autograd forces from a model's energy and verify them against
      central differences.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    cutoff = mo.ui.slider(start=2.0, stop=6.0, step=0.1, value=5.0, label="cutoff (Å)")
    cutoff
    return (cutoff,)


@app.cell(hide_code=True)
def _(mo):
    n_layers = mo.ui.slider(start=1, stop=5, step=1, value=3, label="n_layers")
    n_layers
    return (n_layers,)


@app.cell
def _(cutoff):
    from tinymlip import build_graph, load_rmd17

    bundle = load_rmd17("ethanol", split="train", n_frames=1, seed=0)
    atoms = bundle.structures[0]

    import torch

    torch.manual_seed(0)
    graph = build_graph(atoms, cutoff=cutoff.value)
    graph
    return atoms, build_graph, graph, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the pipeline does, line by line

    Every modern MLIP runs roughly this four-step pipeline. The next cell walks
    it by hand so you can watch each step's output:

    1. **Embed atomic numbers.** Each atom starts as a learnable feature
       vector keyed by its element. Untrained, so the vectors are random —
       but every H atom starts *identical* to every other H, every C
       identical to every other C. Element is the only thing the model
       "knows" before message passing.
    2. **Message passing.** Each interaction layer mixes in features from
       neighbors (notebook 02). After $k$ layers, atom $i$ has seen
       everything within $k$ hops on the graph. Two H atoms that started
       identical become *distinguishable* the moment their neighborhoods
       differ.
    3. **Per-atom readout.** A small MLP turns each atom's final feature
       vector into one scalar — its per-atom energy contribution. The MLP
       is shared across atoms; the same function gets applied independently
       to each one.
    4. **Sum.** The molecular energy is the sum of those per-atom scalars.
       That single choice is what makes the energy size-extensive (the next
       section verifies it).

    Below, `atom_embeddings`, `atom_features`, `per_atom_energies`, and
    `energy` are the intermediate tensors at each step.
    """)
    return


@app.cell
def _(cutoff, graph, mo, n_layers, torch):
    from ase.data import chemical_symbols

    from tinymlip import InvariantMPNN

    torch.manual_seed(0)
    model = InvariantMPNN(
        hidden_dim=32,                # features per atom (the size of each atom's feature vector — this is what "F" means in the shape comments below)
        num_basis=8,
        cutoff=cutoff.value,
        n_layers=n_layers.value,
    )

    # Step through the model by hand so the intermediate shapes are visible.
    # (model(graph) computes the same energy in one call — we unroll it here to teach.)

    # 1) Embed: each atomic number becomes a learnable feature vector.
    atom_embeddings = model.embed(graph.z)              # [n_atoms, hidden_dim]

    # 2) Message passing: each layer mixes features from neighbors.
    atom_features = atom_embeddings
    for layer in model.interactions:
        atom_features = layer(atom_features, graph)     # [n_atoms, hidden_dim]

    # 3) Readout: a per-atom MLP turns each feature vector into a scalar.
    per_atom_energies = model.readout(atom_features).squeeze(-1)   # [n_atoms]

    # 4) Sum: the molecular energy is the sum of per-atom contributions.
    energy = per_atom_energies.sum()                    # []  scalar molecular energy

    labels = [
        f"{chemical_symbols[int(graph.z[i])]}[{i}]"
        for i in range(graph.n_atoms)
    ]
    e_total = energy.item()

    mo.md(
        f"**Pipeline shapes** (one row per step):\n\n"
        f"- embedding(z): `{tuple(atom_embeddings.shape)}`\n"
        f"- after {n_layers.value} interaction(s): `{tuple(atom_features.shape)}`\n"
        f"- readout: `{tuple(per_atom_energies.shape)}`\n"
        f"- sum: `{tuple(energy.shape)}` → scalar **E_total = {e_total:.4f}**"
    )
    return (
        InvariantMPNN,
        chemical_symbols,
        e_total,
        labels,
        model,
        per_atom_energies,
    )


@app.cell(hide_code=True)
def _(e_total, graph, labels, n_layers, per_atom_energies):
    import plotly.graph_objects as go

    fig_per_atom = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=per_atom_energies.detach().numpy(),
                marker_color="#1f77b4",
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            )
        ]
    )
    fig_per_atom.add_hline(
        y=e_total / graph.n_atoms,
        line=dict(dash="dash", color="grey"),
        annotation_text=f"mean = E_total / N = {e_total / graph.n_atoms:.3f}",
        annotation_position="top right",
    )
    fig_per_atom.update_layout(
        title=(
            f"Per-atom energies for ethanol "
            f"(untrained model, h=32, layers={n_layers.value}). "
            f"Sum: E_total = {e_total:.4f}"
        ),
        xaxis_title="atom",
        yaxis_title="per-atom energy (arbitrary units)",
        height=360,
    )
    fig_per_atom
    return (go,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading this chart

    Each bar is one atom's contribution to the total energy. The dashed line
    is `E_total / N` — the mean — and the molecular energy is the sum of all
    the bars (nothing else combines them).

    A few things to notice (defaults: cutoff = 5 Å, `n_layers = 3`):

    - **Absolute values aren't meaningful.** The model is untrained, so the
      scale is arbitrary. What's interesting is the *shape* — which atoms
      contribute more, and which cluster together.
    - **Atoms in the same chemical environment cluster.** All H atoms
      started identical in the embedding; any difference you see comes from
      message passing. In this rMD17 ethanol structure the heavy atoms are
      `C[0]` (the carbon bonded to oxygen), `C[1]` (the methyl carbon), and
      `O[2]`. So:
        - `H[3]`, `H[4]` are **methylene** H's (on `C[0]`),
        - `H[5]`, `H[6]`, `H[7]` are **methyl** H's (on `C[1]`),
        - `H[8]` is the **hydroxyl** H (on `O[2]`).
      At the default settings you can see the three groups split apart on
      the chart even though no training has happened — message passing
      alone is enough to make chemically distinct hydrogens distinguishable.
    - **Move the `n_layers` slider.** At `n_layers = 1`, the H bars nearly
      collapse — one layer of message passing only reaches the immediate
      neighbor, and every H has just one heavy-atom neighbor, so the model
      can barely tell them apart. At `n_layers = 5`, the random init gets
      amplified through the extra stacks and the clean three-group picture
      blurs. `n_layers = 3` is the sweet spot for this untrained demo;
      after training, deeper models would tell a cleaner story.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why a sum?

    The molecular energy is the *sum* of per-atom contributions. That single
    choice gives the model a property real molecules have for free:
    **size-extensivity** — twice as much molecule, twice as much energy.

    A model that pooled the per-atom features into one global vector and then
    passed it through a final MLP would NOT be size-extensive. Two copies of
    ethanol sitting 20 Å apart should give exactly twice the energy of one — and
    because the radial cutoff guarantees no edges between the copies, our sum
    delivers that automatically (no training required).
    """)
    return


@app.cell
def _(atoms, build_graph, cutoff, graph, mo, model, torch):
    import ase
    import numpy as np

    # Build two copies of ethanol, 20 Å apart along x. At cutoff = 5 Å, no edges
    # connect the copies — they're independent.
    positions_single = atoms.get_positions()                    # [n_atoms, 3]
    dimer_offset = np.array([20.0, 0.0, 0.0])                   # [3]
    positions_dimer = np.concatenate(
        [positions_single, positions_single + dimer_offset],
        axis=0,
    )                                                           # [2*n_atoms, 3]
    numbers_dimer = np.concatenate(
        [atoms.numbers, atoms.numbers],
        axis=0,
    )                                                           # [2*n_atoms]
    atoms_dimer = ase.Atoms(numbers=numbers_dimer, positions=positions_dimer)
    graph_dimer = build_graph(atoms_dimer, cutoff=cutoff.value)

    with torch.no_grad():
        energy_single = model(graph).item()
        energy_dimer = model(graph_dimer).item()

    ratio = energy_dimer / energy_single if abs(energy_single) > 1e-12 else float("nan")
    extensivity_residual = energy_dimer - 2 * energy_single

    mo.md(
        f"**E(ethanol)** = `{energy_single:.6f}`\n\n"
        f"**E(two ethanols, 20 Å apart)** = `{energy_dimer:.6f}`\n\n"
        f"**ratio E_dimer / E_single** = `{ratio:.6f}` (target: exactly 2)\n\n"
        f"**E_dimer − 2·E_single** = `{extensivity_residual:.2e}`\n\n"
        f"That residual is essentially zero (machine precision in float32). The "
        f"sum readout + finite cutoff make this *structural*, not learned — it "
        f"holds on the untrained model and would still hold after training."
    )
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forces from autograd

    The molecular energy is a scalar function of every atom's position. The
    force on atom $i$ is the negative gradient of that energy with respect to
    $\mathbf{r}_i$:

    $$\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}.$$

    Two things follow for free:

    1. **The forces are conservative.** Anything that comes out of $-\nabla E$
       has zero curl, so the line integral around a closed loop in position
       space is zero. In molecular dynamics that means the NVE ensemble
       conserves energy — a property real force fields must have and that
       *separately predicted* forces don't generally satisfy.
    2. **`torch.autograd.grad` does the bookkeeping.** As long as `graph.pos`
       is the autograd leaf (we set `requires_grad=True` on it before the
       forward pass), one autograd call gives us all 3N force components.

    We package that one line as `tinymlip.compute_forces(energy, pos)`. The
    forces below are from our **untrained** model, so their magnitudes and
    directions are random — what matters is that they exist, they have the
    right shape, and Σ F ≈ 0 (which we'll check explicitly in the next
    section).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    arrow_scale = mo.ui.slider(
        start=1.0, stop=50.0, step=1.0, value=20.0,
        label="arrow length scale",
    )
    arrow_scale
    return (arrow_scale,)


@app.cell
def _(atoms, build_graph, cutoff, mo, model):
    from tinymlip import compute_forces

    # To compute forces from autograd, the positions must be the autograd leaf:
    # set requires_grad on `pos` BEFORE the forward pass. We rebuild the graph
    # here rather than mutating the upstream `graph` so the earlier cells stay
    # clean.
    graph_with_grad = build_graph(atoms, cutoff=cutoff.value)
    graph_with_grad.pos.requires_grad_(True)

    predicted_energy = model(graph_with_grad)                          # scalar
    forces = compute_forces(predicted_energy, graph_with_grad.pos)     # [n_atoms, 3]

    mo.md(
        f"`forces.shape` = `{tuple(forces.shape)}` &nbsp;·&nbsp; "
        f"`|F|_max` = `{forces.detach().norm(dim=-1).max().item():.4f}` "
        f"(arbitrary units — model is untrained)"
    )
    return compute_forces, forces, graph_with_grad, predicted_energy


@app.cell(hide_code=True)
def _(
    arrow_scale,
    atoms,
    build_graph,
    chemical_symbols,
    forces,
    go,
    graph_with_grad,
    n_layers,
):
    from tinymlip.viz import element_color, element_radius

    positions_np = graph_with_grad.pos.detach().numpy()                # [n_atoms, 3]
    forces_np = forces.detach().numpy()                                # [n_atoms, 3]
    atomic_numbers_np = graph_with_grad.z.numpy()                      # [n_atoms]

    # Bonds: a separate, tighter-cutoff graph just for visualization. The physics
    # graph above uses `cutoff` (default 5 Å), which on ethanol is fully connected
    # (72 edges) — drawing those would obscure the structure. A 1.6 Å cutoff
    # isolates the covalent bonds (same trick as notebook 02).
    bond_cutoff = 1.6
    graph_bonds = build_graph(atoms, cutoff=bond_cutoff)
    bond_src, bond_dst = graph_bonds.edge_index
    bond_x, bond_y, bond_z = [], [], []
    for s_k, d_k in zip(bond_src.tolist(), bond_dst.tolist(), strict=True):
        if s_k < d_k:
            bond_x.extend([positions_np[s_k, 0], positions_np[d_k, 0], None])
            bond_y.extend([positions_np[s_k, 1], positions_np[d_k, 1], None])
            bond_z.extend([positions_np[s_k, 2], positions_np[d_k, 2], None])

    arrow_length_scale = float(arrow_scale.value)
    arrow_tail = positions_np
    arrow_tip = positions_np + arrow_length_scale * forces_np          # [n_atoms, 3]

    shaft_x, shaft_y, shaft_z = [], [], []
    for k in range(graph_with_grad.n_atoms):
        shaft_x.extend([arrow_tail[k, 0], arrow_tip[k, 0], None])
        shaft_y.extend([arrow_tail[k, 1], arrow_tip[k, 1], None])
        shaft_z.extend([arrow_tail[k, 2], arrow_tip[k, 2], None])

    force_magnitudes = forces.detach().norm(dim=-1).numpy()            # [n_atoms]
    head_u = forces_np[:, 0] * arrow_length_scale
    head_v = forces_np[:, 1] * arrow_length_scale
    head_w = forces_np[:, 2] * arrow_length_scale

    fig_forces = go.Figure()
    fig_forces.add_trace(
        go.Scatter3d(
            x=bond_x, y=bond_y, z=bond_z,
            mode="lines",
            line=dict(color="#bbbbbb", width=3),
            hoverinfo="skip",
            showlegend=False,
            name="bonds",
        )
    )
    fig_forces.add_trace(
        go.Scatter3d(
            x=shaft_x, y=shaft_y, z=shaft_z,
            mode="lines",
            line=dict(color="crimson", width=4),
            hoverinfo="skip",
            showlegend=False,
            name="forces",
        )
    )
    fig_forces.add_trace(
        go.Cone(
            x=arrow_tip[:, 0], y=arrow_tip[:, 1], z=arrow_tip[:, 2],
            u=head_u, v=head_v, w=head_w,
            anchor="tail",
            sizemode="absolute",
            sizeref=0.08,
            colorscale=[[0, "crimson"], [1, "crimson"]],
            showscale=False,
            hoverinfo="skip",
            name="arrowheads",
        )
    )
    fig_forces.add_trace(
        go.Scatter3d(
            x=positions_np[:, 0], y=positions_np[:, 1], z=positions_np[:, 2],
            mode="markers+text",
            marker=dict(
                size=[element_radius(int(zk)) * 14 for zk in atomic_numbers_np],
                color=[element_color(int(zk)) for zk in atomic_numbers_np],
                line=dict(color="#222", width=1),
            ),
            text=[
                f"{chemical_symbols[int(zk)]}[{k}]"
                for k, zk in enumerate(atomic_numbers_np)
            ],
            textposition="top center",
            textfont=dict(size=10, color="#111"),
            hovertemplate=[
                f"atom {k} ({chemical_symbols[int(zk)]}): |F| = {force_magnitudes[k]:.4f}<extra></extra>"
                for k, zk in enumerate(atomic_numbers_np)
            ],
            showlegend=False,
            name="atoms",
        )
    )

    max_force_magnitude = float(force_magnitudes.max())
    fig_forces.update_layout(
        title=(
            f"Autograd forces on ethanol (untrained, h=32, layers={n_layers.value}). "
            f"|F|_max = {max_force_magnitude:.4f} · arrows scaled by {arrow_length_scale:.0f}"
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            dragmode="turntable",
        ),
        height=480,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig_forces
    return


@app.cell
def _(
    atoms,
    build_graph,
    compute_forces,
    cutoff,
    forces,
    mo,
    model,
    np,
    predicted_energy,
):
    # Two structural properties to check:
    #   (a) E is translation-invariant: shifting every atom by the same vector
    #       must leave the energy unchanged.
    #   (b) Sum of forces is zero: Newton's third law / global momentum
    #       conservation. Both follow from the energy depending only on
    #       inter-atomic distances (and the cutoff envelope) — neither is
    #       trained, both are baked into the architecture.

    shift_vector = np.array([1.2, -0.7, 2.4])                       # [3]
    atoms_shifted = atoms.copy()
    atoms_shifted.set_positions(atoms_shifted.get_positions() + shift_vector)

    graph_shifted = build_graph(atoms_shifted, cutoff=cutoff.value)
    graph_shifted.pos.requires_grad_(True)
    energy_shifted = model(graph_shifted)                           # scalar
    forces_shifted = compute_forces(energy_shifted, graph_shifted.pos)  # [n_atoms, 3]

    energy_diff = (energy_shifted - predicted_energy).abs().item()
    force_sum_original = forces.detach().sum(dim=0).norm().item()
    force_sum_shifted = forces_shifted.detach().sum(dim=0).norm().item()

    mo.md(
        f"### Translation invariance\n\n"
        f"Shifting all positions by `{shift_vector.tolist()}` Å:\n\n"
        f"- `|E(shifted) − E(original)|` = `{energy_diff:.2e}` ✅ (should be ~0)\n\n"
        f"### Force conservation (Σ F ≈ 0)\n\n"
        f"- original geometry: `||Σ F_i||` = `{force_sum_original:.2e}` ✅\n"
        f"- shifted geometry:  `||Σ F_i||` = `{force_sum_shifted:.2e}` ✅\n\n"
        f"Both are at float32 roundoff. Translation invariance holds because "
        f"the model only ever sees `pos[dst] − pos[src]` differences (recomputed "
        f"inside each interaction layer). Σ F = 0 then follows by Noether: "
        f"translation symmetry ⇔ conservation of total momentum."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical gradient check

    If `compute_forces` really computes $-\nabla E$, then central differences
    on the energy should agree with the autograd-derived force at the same
    atom and axis. We pick an atom and an axis, perturb the position by
    $\pm \varepsilon$ Å, and compare:

    $$\text{numerical} = -\,\frac{E(\mathbf{r} + \varepsilon \hat{\mathbf{e}}_{i,\alpha}) - E(\mathbf{r} - \varepsilon \hat{\mathbf{e}}_{i,\alpha})}{2\varepsilon}.$$

    We use **float64** for both the model and the graph here — the
    central-difference truncation error is $O(\varepsilon^2)$, so float32
    roundoff would dominate at $\varepsilon = 10^{-3}$ Å. (The same
    comparison runs in the test suite under `tests/test_forces.py`.)
    """)
    return


@app.cell(hide_code=True)
def _(graph, mo):
    atom_idx = mo.ui.dropdown(
        options={str(i): i for i in range(graph.n_atoms)},
        value="0",
        label="atom index",
    )
    atom_idx
    return (atom_idx,)


@app.cell(hide_code=True)
def _(mo):
    axis_idx = mo.ui.dropdown(
        options={"x": 0, "y": 1, "z": 2},
        value="x",
        label="axis",
    )
    axis_idx
    return (axis_idx,)


@app.cell
def _(
    InvariantMPNN,
    atom_idx,
    atoms,
    axis_idx,
    build_graph,
    chemical_symbols,
    compute_forces,
    cutoff,
    graph,
    mo,
    n_layers,
    torch,
):
    # Build a float64 model with the same architecture as the float32 one
    # upstream. The property under test, F = -grad(E), holds for any model;
    # float64 is here so central-difference truncation error dominates the
    # comparison instead of float32 roundoff.
    torch.manual_seed(0)
    model_fp64 = InvariantMPNN(
        hidden_dim=32,
        num_basis=8,
        cutoff=cutoff.value,
        n_layers=n_layers.value,
    ).double()

    eps = 1e-3
    atom_index = atom_idx.value
    axis_index = axis_idx.value

    # 1) Autograd force at (atom_index, axis_index).
    graph_fp64 = build_graph(atoms, cutoff=cutoff.value, dtype=torch.float64)
    graph_fp64.pos.requires_grad_(True)
    energy_fp64 = model_fp64(graph_fp64)                            # scalar
    forces_fp64 = compute_forces(energy_fp64, graph_fp64.pos)       # [n_atoms, 3]
    autograd_force = forces_fp64[atom_index, axis_index].item()

    # 2) Central differences. Rebuild the graph each time so connectivity is
    #    always recomputed from the perturbed positions (won't change at this
    #    eps, but the pattern matches the pytest version under
    #    tests/test_forces.py).
    def _energy_at_offset(delta):
        atoms_perturbed = atoms.copy()
        perturbed_positions = atoms_perturbed.get_positions()
        perturbed_positions[atom_index, axis_index] += delta
        atoms_perturbed.set_positions(perturbed_positions)
        graph_perturbed = build_graph(
            atoms_perturbed, cutoff=cutoff.value, dtype=torch.float64,
        )
        with torch.no_grad():
            return model_fp64(graph_perturbed).item()

    energy_plus = _energy_at_offset(+eps)
    energy_minus = _energy_at_offset(-eps)
    numerical_force = -(energy_plus - energy_minus) / (2 * eps)
    absolute_error = abs(numerical_force - autograd_force)

    axis_label = {0: "x", 1: "y", 2: "z"}[axis_index]
    element_label = chemical_symbols[int(graph.z[atom_index])]
    mo.md(
        f"With `ε = {eps}` Å, atom **{atom_index}** ({element_label}), "
        f"axis **{axis_label}**:\n\n"
        f"- autograd `F[{atom_index}, {axis_label}]`     = `{autograd_force:+.8f}`\n\n"
        f"- numerical `−(E₊ − E₋)/(2ε)` = `{numerical_force:+.8f}`\n\n"
        f"- `|autograd − numerical|`     = `{absolute_error:.2e}`\n\n"
        f"The gap is dominated by the central-difference truncation error "
        f"`O(ε²) ≈ 10⁻⁶`, not by autograd noise. **F = −∇E** holds to "
        f"machine precision, exactly as it should."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we deferred (and where it goes)

    We've built an end-to-end MLIP: embedding → message passing → readout →
    sum → autograd forces. Three things on purpose did not show up here:

    - **Training.** The model in every cell above is *untrained* — its
      forces are random but valid (conservative, translation-invariant,
      summing to zero). Notebook 04 trains `InvariantMPNN` on rMD17 with an
      energy + force-matching loss.
    - **Equivariant message passing.** Our scalar messages discard the
      *direction* an edge points. For many physical observables (forces,
      dipoles, polarizabilities) that's a real handicap. Notebook 05
      introduces `EquivariantMPNN` — same anatomy, but each atom also
      carries vector channels that rotate with the molecule — and compares
      the two side-by-side on the same training run.
    - **Periodic systems.** Real materials live in periodic cells. Notebook
      06 adds PBC support to the neighbor list and demos the model as an
      ASE calculator on a small crystal.
    """)
    return


if __name__ == "__main__":
    app.run()
