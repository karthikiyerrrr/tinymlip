import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 · Training the invariant MPNN

    **What this notebook teaches.** Notebook 03 built an untrained MLIP whose
    forces were random scribbles. This notebook trains it: the same
    `InvariantMPNN` learns from rMD17 reference energies and forces, and by
    the end its predicted forces should point along bonds with the right
    magnitudes.

    The arc:

    1. **Why train an MLIP at all.** DFT's cost scales roughly cubically with
       system size (and worse for the more accurate variants). An MLIP forward
       pass does a constant amount of work per atom per layer, and edges
       scale linearly under a fixed cutoff — so inference is O(N). Training
       trades one expensive batch of DFT calculations for a model that is
       then cheap to evaluate forever.
    2. **Batching.** We collate many molecules into one big disjoint-union
       graph so the model sees a whole batch in a single forward pass.
    3. **Per-element reference shift.** rMD17 ethanol energies sit in the
       tens-of-thousands of kcal/mol. Subtracting a per-element offset turns
       the targets into small residuals (~kcal/mol), which are far easier to
       learn.
    4. **Energy + force loss.** A weighted MSE on per-atom energy and on
       force components. Forces still come from autograd through the energy
       (no separate head) — same conservation guarantee as notebook 03.
    5. **The loop itself.** For each epoch: train, evaluate on a held-out
       validation set, log to a polars DataFrame.
    6. **Look at the forces.** Same molecule as notebook 03, same arrow plot,
       but the arrows now point along bonds.

    **Prerequisites.** Notebooks 01–03.

    **By the end you can:**
    - Run a small training loop end-to-end on rMD17 in under five minutes on CPU.
    - Read learning curves and parity plots without confusion.
    - Explain what the per-element reference shift is doing and why.

    **A note on the sliders.** Model architecture (`hidden_dim`, `n_layers`,
    `num_basis`) is fixed in this notebook so the sliders only control the
    training story — learning rate, epochs, batch size, and the force-loss
    weight `w_F`. Going bigger or deeper is a one-line change in the `tiny`
    dict below, but costs more time on CPU.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import polars as pl
    import torch
    from torch.utils.data import DataLoader

    from tinymlip import (
        InvariantMPNN,
        apply_atomic_reference,
        build_graph,
        collate_graphs,
        compute_forces,
        fit_atomic_reference,
        load_rmd17,
        make_collate,
        to_torch_dataset,
        train,
    )

    torch.manual_seed(0)
    np.random.seed(0)
    return (
        DataLoader,
        InvariantMPNN,
        apply_atomic_reference,
        build_graph,
        collate_graphs,
        compute_forces,
        fit_atomic_reference,
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
    lr = mo.ui.slider(
        steps=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        value=1e-3,
        label="learning rate",
        show_value=True,
    )
    n_epochs = mo.ui.slider(start=5, stop=60, step=5, value=30, label="n_epochs", show_value=True)
    batch_size = mo.ui.slider(
        start=2, stop=32, step=2, value=8, label="batch size", show_value=True
    )
    w_f = mo.ui.slider(
        start=1.0, stop=200.0, step=1.0, value=100.0, label="w_F (force weight)", show_value=True
    )

    mo.vstack([lr, n_epochs, batch_size, w_f])
    return batch_size, lr, n_epochs, w_f


@app.cell
def _():
    # `tiny` preset — target: under 5 min on CPU. Sliders above override lr,
    # n_epochs, batch_size, and w_f; the rest stay fixed so the notebook is a
    # story about training, not about hyperparameter search.
    tiny = {
        "n_train": 500,
        "n_val": 100,
        "n_test": 100,
        "hidden_dim": 32,
        "num_basis": 8,
        "n_layers": 3,
        "cutoff": 5.0,
        "w_e": 1.0,
    }
    tiny
    return (tiny,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load rMD17 ethanol

    rMD17 is the *revised* MD17 dataset (Christensen & von Lilienfeld 2020) —
    the same MD trajectories as MD17 but recomputed at a more accurate level
    of DFT, so the labels are cleaner. We use the official 5-fold CV split #1
    and subsample to keep the run inside the five-minute CPU budget.

    `load_rmd17` returns an `RMD17Bundle` with two parallel views: a polars
    `meta` table (one row per frame) and `structures`, a list of ASE Atoms
    with positions, atomic numbers, and per-atom forces. The energy lives
    in both — `meta["energy"]` for tabular access, `atoms.info["energy"]`
    for the per-frame view.

    **Units.** rMD17 ships in kcal/mol for energies and kcal/mol/Å for force
    components, and we keep those throughout the notebook. If you prefer eV
    and eV/Å (common in the broader MLIP literature), multiply by ≈ 0.0434.
    """)
    return


@app.cell
def _(load_rmd17, mo, tiny):
    # rMD17 has separate train and test splits. We further carve a held-out
    # validation slice off the *training* split (last n_val frames) so the
    # official test split stays untouched until the parity plots at the end.
    trainval_bundle = load_rmd17(
        "ethanol",
        split="train",
        cv_fold=1,
        n_frames=tiny["n_train"] + tiny["n_val"],
        seed=0,
    )
    test_bundle = load_rmd17(
        "ethanol",
        split="test",
        cv_fold=1,
        n_frames=tiny["n_test"],
        seed=0,
    )

    n_train = tiny["n_train"]
    train_structures = trainval_bundle.structures[:n_train]
    val_structures = trainval_bundle.structures[n_train:]
    train_meta = trainval_bundle.meta.head(n_train)
    val_meta = trainval_bundle.meta.slice(n_train)

    mo.md(
        f"**train:** {len(train_structures)} frames &nbsp;·&nbsp; "
        f"**val:** {len(val_structures)} frames &nbsp;·&nbsp; "
        f"**test:** {len(test_bundle.structures)} frames\n\n"
        "**On the split discipline.** rMD17 is an MD trajectory — consecutive "
        "frames are autocorrelated on the picosecond timescale of bond "
        "vibrations, so naive uniform random subsampling would bleed training "
        "frames into the test set. The official 5-fold CV indices are "
        "designed to be temporally separated for exactly this reason. We "
        "carve validation out of the *training* split so the official test "
        "split stays untouched until the parity plots at the end. (The "
        "analog in notebook 06’s crystal data is: don’t put the same "
        "crystal at different volumes in train and test.)"
    )
    return test_bundle, train_meta, train_structures, val_meta, val_structures


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-element reference shift

    Raw rMD17 ethanol energies sit around −97 000 kcal/mol — totally
    dominated by the per-atom "chemical reference" contribution that any MLIP
    would otherwise spend its capacity learning. The standard trick (mirroring
    SchNetPack's `RemoveOffsets` / `AddOffsets`) is to fit a per-element
    offset on the training set and *subtract* it from every target before
    training, so the model only ever sees small *residuals* (~kcal/mol).
    This is the $E_0(z)$ shift notebook 03 forward-referenced.

    The fit is a one-line linear regression:

    $$
    E_{\text{frame}} \;\approx\; \sum_{z} n_z(\text{frame}) \cdot s_z,
    $$

    where $n_z(\text{frame})$ counts atoms of element $z$ in the frame and
    $s_z$ is the per-element offset we learn. `fit_atomic_reference` solves
    this via `np.linalg.lstsq` and returns `{z: s_z}`.

    **A caveat on what these numbers mean.** Every ethanol frame has
    composition C₂H₆O, so every row of the design matrix is `[2, 6, 1]` —
    the linear system is rank-1, and `np.linalg.lstsq` returns the
    minimum-norm solution. Only the composition-weighted sum
    `2·s_C + 6·s_H + s_O` is identified by the data; the individual values
    shouldn't be read as physical per-element binding energies. They become
    physically meaningful when the dataset spans multiple compositions
    (notebook 06's crystals).

    **The shift does not bias force learning.** It depends on atomic
    numbers only, not positions, so $\partial(\text{shift}) / \partial \mathbf{r} = 0$.
    Forces are unaffected; only the energy target gets shifted. This is
    also why the `w_E = 1`, `w_F = 100` balance below lands cleanly — the
    shift makes the energy loss well-conditioned without touching the force
    loss at all.
    """)
    return


@app.cell
def _(fit_atomic_reference, mo, np, train_meta, train_structures):
    from ase.data import chemical_symbols

    train_energies = train_meta["energy"].to_numpy()
    shifts = fit_atomic_reference(train_structures, train_energies)

    # Apply the shift to every frame to get the residuals the model will learn.
    def _residual(structures, energies):
        out = []
        for atoms, e in zip(structures, energies, strict=True):
            offset = sum(shifts[int(z)] for z in atoms.numbers)
            out.append(e - offset)
        return np.array(out)

    train_residuals = _residual(train_structures, train_energies)

    raw_span = float(train_energies.max() - train_energies.min())
    res_span = float(train_residuals.max() - train_residuals.min())
    shifts_md = "\n".join(
        f"- `{chemical_symbols[z]}`: {s:+.4f} kcal/mol per atom" for z, s in sorted(shifts.items())
    )
    mo.md(
        f"**Fitted per-element offsets:**\n\n"
        f"{shifts_md}\n\n"
        f"**Raw E** range: `[{train_energies.min():.1f}, {train_energies.max():.1f}]` "
        f"kcal/mol (span {raw_span:.2f})\n\n"
        f"**Residual E − Σshift(z)** range: "
        f"`[{train_residuals.min():.4f}, {train_residuals.max():.4f}]` "
        f"kcal/mol (span {res_span:.4f})\n\n"
        f"Same data, four orders of magnitude smaller numbers."
    )
    return shifts, train_energies, train_residuals


@app.cell(hide_code=True)
def _(train_energies, train_residuals):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig_shift = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Raw E (train)", "E − Σ shift(z)  (train)"),
        horizontal_spacing=0.12,
    )
    fig_shift.add_trace(
        go.Histogram(x=train_energies, marker_color="#888", nbinsx=40, showlegend=False),
        row=1,
        col=1,
    )
    fig_shift.add_trace(
        go.Histogram(x=train_residuals, marker_color="#1f77b4", nbinsx=40, showlegend=False),
        row=1,
        col=2,
    )
    fig_shift.update_xaxes(title_text="energy (kcal/mol)", row=1, col=1)
    fig_shift.update_xaxes(title_text="residual (kcal/mol)", row=1, col=2)
    fig_shift.update_yaxes(title_text="count", row=1, col=1)
    fig_shift.update_layout(
        title="Reference shift collapses the targets onto an O(1) scale",
        height=320,
        margin=dict(l=10, r=10, t=70, b=40),
        bargap=0.05,
    )
    fig_shift
    return go, make_subplots


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Batching: many molecules, one graph

    Training one molecule at a time is correct but slow. The standard MLIP
    trick is the **disjoint union**: stitch many graphs together into one big
    graph, keeping a `batch` vector that says which atom belongs to which
    frame. Because each original graph's edges only connect *its own* atoms,
    the union has no edges crossing between frames — a single forward pass
    through the model handles the whole batch.

    We expose this as `collate_graphs(list[AtomGraph]) -> AtomGraph`. The
    `make_collate(cutoff)` helper wires it into a `torch.utils.data.DataLoader`
    collate function. Below, four ethanol graphs (9 atoms each) become one
    batched graph of 36 atoms with `batch = [0,0,...,1,1,...,2,2,...,3,3,...]`.
    """)
    return


@app.cell
def _(build_graph, collate_graphs, mo, tiny, train_structures):
    # Build four graphs and disjoint-union them.
    demo_graphs = [build_graph(train_structures[i], cutoff=tiny["cutoff"]) for i in range(4)]
    batched_demo = collate_graphs(demo_graphs)

    mo.md(
        f"**per-frame:** {[g.n_atoms for g in demo_graphs]} atoms, "
        f"{[g.n_edges for g in demo_graphs]} edges\n\n"
        f"**after collate:** `n_atoms = {batched_demo.n_atoms}`, "
        f"`n_edges = {batched_demo.n_edges}`\n\n"
        f"**batch:** `{batched_demo.batch.tolist()}`"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The loss

    We train against both energies AND forces, weighting them so neither
    term dominates:

    $$
    \mathcal{L} \;=\; w_E \cdot \operatorname{MSE}\!\left(\frac{E_{\text{pred}}}{N},\, \frac{E_{\text{true}}}{N}\right)
    \;+\; w_F \cdot \operatorname{MSE}\!\bigl(\mathbf{F}_{\text{pred}},\, \mathbf{F}_{\text{true}}\bigr).
    $$

    - **Per-atom normalization on energy.** Without it, larger systems contribute
      more to the loss simply because they have more atoms. Notebook 06 mixes
      9-atom ethanols with much larger crystals, where this matters in earnest.
    - **Why w_F default = 100.** Post-shift energies span ~1 kcal/mol; force
      components span ~50 kcal/mol/Å. Squared, those scale ratios are ~2 500
      apart, so a force weight of 100 brings the two MSEs into the same
      ballpark.

    `tinymlip.train.energy_force_loss` returns `(loss_tensor, metrics_dict)`.
    The dict reports MAE in human-readable units alongside the scalar loss.

    **`w_F = 100` is a starting point, not a derived truth.** Even after
    MSE-scale balancing, the two terms can still compete — improving the
    force fit can hurt the energy fit and vice versa. Production MLIPs
    treat `w_F` as a tuned hyperparameter, typically in the range
    **10–1000**, and usually force-dominant because forces are what drive
    the downstream MD simulation. The `w_F` slider above lets you watch
    the trade-off in real time.

    <details>
    <summary>Try it: set <code>w_F = 1</code>, then <code>w_F = 200</code>. What happens to energy MAE vs. force MAE over 30 epochs?</summary>

    At `w_F = 1` the energy term overwhelmingly dominates the loss: energy
    MAE drops further than at `w_F = 100`, but force MAE plateaus at a
    much higher value — the optimizer effectively ignores forces because
    their contribution to the loss is now about `1 / 2500` of the energy
    contribution. At `w_F = 200` you see the opposite: force MAE drops
    faster, energy MAE plateaus higher. That's the trade-off curve —
    which point on it is "best" depends on what you're going to do with
    the trained model.
    </details>
    """)
    return


@app.cell
def _(
    DataLoader,
    batch_size,
    make_collate,
    mo,
    test_bundle,
    tiny,
    to_torch_dataset,
    train_meta,
    train_structures,
    val_meta,
    val_structures,
):
    # Wrap the ASE Atoms lists in lightweight polars-backed bundles so
    # `to_torch_dataset` can yield per-frame dicts.
    from tinymlip.data import RMD17Bundle

    train_bundle_split = RMD17Bundle(meta=train_meta, structures=list(train_structures))
    val_bundle_split = RMD17Bundle(meta=val_meta, structures=list(val_structures))

    collate = make_collate(cutoff=tiny["cutoff"])
    train_loader = DataLoader(
        to_torch_dataset(train_bundle_split),
        batch_size=batch_size.value,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        to_torch_dataset(val_bundle_split),
        batch_size=batch_size.value,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        to_torch_dataset(test_bundle),
        batch_size=batch_size.value,
        shuffle=False,
        collate_fn=collate,
    )

    mo.md(
        f"`train_loader`: **{len(train_loader)}** batches of size {batch_size.value} "
        f"(last may be shorter)\n\n"
        f"`val_loader`: **{len(val_loader)}** batches\n\n"
        f"`test_loader`: **{len(test_loader)}** batches"
    )
    return test_loader, train_loader, val_loader


@app.cell
def _(
    InvariantMPNN,
    apply_atomic_reference,
    compute_forces,
    mo,
    shifts,
    tiny,
    torch,
    train_loader,
    w_f,
):
    from tinymlip.train import energy_force_loss

    # Build the model (untrained). We will re-instantiate it for real training
    # below so a slider change starts from fresh weights.
    torch.manual_seed(0)
    sanity_model = InvariantMPNN(
        hidden_dim=tiny["hidden_dim"],
        num_basis=tiny["num_basis"],
        cutoff=tiny["cutoff"],
        n_layers=tiny["n_layers"],
    )

    # One untrained forward on one batch from the training set.
    _sanity_batch = next(iter(train_loader))
    _g = _sanity_batch["graph"]
    _g.pos.requires_grad_(True)

    _pred_e = sanity_model(_g)
    _pred_f = compute_forces(_pred_e.sum(), _g.pos)

    _ref = apply_atomic_reference(_g.z, _g.batch, shifts).to(_sanity_batch["energy"].dtype)
    _true_residual = _sanity_batch["energy"] - _ref

    _, sanity_metrics = energy_force_loss(
        _pred_e,
        _true_residual,
        _pred_f,
        _sanity_batch["forces"],
        _sanity_batch["n_atoms"],
        w_e=tiny["w_e"],
        w_f=w_f.value,
    )

    mo.md(
        f"### Untrained-model sanity numbers\n\n"
        f"On one batch from the training set, BEFORE any training:\n\n"
        f"- `loss` = `{sanity_metrics['loss']:.4f}`\n"
        f"- per-atom `energy_mae` = `{sanity_metrics['energy_mae']:.4f}` kcal/mol/atom\n"
        f"- `force_mae` = `{sanity_metrics['force_mae']:.4f}` kcal/mol/Å\n\n"
        f"These should be large — the model is initialized randomly, so its energies "
        f"have no relation to ethanol. The training loop below should pull them down."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The training loop

    `tinymlip.train.train` runs the schedule:

    ```python
    for epoch in range(n_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, ...)
        val_metrics   = evaluate(model, val_loader, ...)
        log[epoch]    = {"train": train_metrics, "val": val_metrics}
    ```

    The optimizer is plain `torch.optim.Adam(lr)`. We're skipping LR schedules,
    EMA, gradient clipping, and other production niceties on purpose — the goal
    is to see the loop, not to chase the last bit of accuracy.

    One thing worth flagging: **`evaluate` cannot use `torch.no_grad()`.**
    Forces are derived via autograd through `graph.pos`, so the autograd graph
    has to stay alive even when we're not updating parameters. We just don't
    call `optimizer.step()`.

    Change a slider above and this cell re-runs end-to-end. On CPU the default
    `tiny` settings should finish in well under a minute.
    """)
    return


@app.cell
def _(
    InvariantMPNN,
    lr,
    mo,
    n_epochs,
    pl,
    shifts,
    tiny,
    torch,
    train,
    train_loader,
    val_loader,
    w_f,
):
    # Fresh model — every slider change starts from the same initial weights.
    torch.manual_seed(0)
    model = InvariantMPNN(
        hidden_dim=tiny["hidden_dim"],
        num_basis=tiny["num_basis"],
        cutoff=tiny["cutoff"],
        n_layers=tiny["n_layers"],
    )

    import time as _time

    _t0 = _time.time()
    history = train(
        model,
        train_loader,
        val_loader,
        n_epochs=n_epochs.value,
        lr=lr.value,
        w_e=tiny["w_e"],
        w_f=w_f.value,
        shifts=shifts,
    )
    wall_seconds = _time.time() - _t0

    _final_train = history.filter(pl.col("split") == "train").tail(1)
    _final_val = history.filter(pl.col("split") == "val").tail(1)
    mo.md(
        f"**Training done in {wall_seconds:.1f}s.**\n\n"
        f"Final train: `loss={float(_final_train['loss'][0]):.4f}`, "
        f"`energy_mae={float(_final_train['energy_mae'][0]):.4f}` kcal/mol/atom, "
        f"`force_mae={float(_final_train['force_mae'][0]):.4f}` kcal/mol/Å\n\n"
        f"Final val: `loss={float(_final_val['loss'][0]):.4f}`, "
        f"`energy_mae={float(_final_val['energy_mae'][0]):.4f}` kcal/mol/atom, "
        f"`force_mae={float(_final_val['force_mae'][0]):.4f}` kcal/mol/Å"
    )
    return history, model


@app.cell(hide_code=True)
def _(go, history, make_subplots, pl):
    fig_curves = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("loss", "MAE"),
        horizontal_spacing=0.12,
    )
    _split_colors = {"train": "#1f77b4", "val": "#ff7f0e"}
    for split in ("train", "val"):
        sub = history.filter(pl.col("split") == split)
        epochs_x = sub["epoch"].to_list()
        fig_curves.add_trace(
            go.Scatter(
                x=epochs_x,
                y=sub["loss"].to_list(),
                mode="lines+markers",
                name=f"{split} loss",
                line=dict(color=_split_colors[split]),
            ),
            row=1,
            col=1,
        )
        fig_curves.add_trace(
            go.Scatter(
                x=epochs_x,
                y=sub["energy_mae"].to_list(),
                mode="lines+markers",
                name=f"{split} energy MAE",
                line=dict(color=_split_colors[split], dash="solid"),
                legendgroup=split,
            ),
            row=1,
            col=2,
        )
        fig_curves.add_trace(
            go.Scatter(
                x=epochs_x,
                y=sub["force_mae"].to_list(),
                mode="lines+markers",
                name=f"{split} force MAE",
                line=dict(color=_split_colors[split], dash="dot"),
                legendgroup=split,
            ),
            row=1,
            col=2,
        )
    fig_curves.update_yaxes(type="log", row=1, col=1)
    fig_curves.update_yaxes(type="log", row=1, col=2)
    fig_curves.update_xaxes(title_text="epoch", row=1, col=1)
    fig_curves.update_xaxes(title_text="epoch", row=1, col=2)
    fig_curves.update_yaxes(title_text="loss (log)", row=1, col=1)
    fig_curves.update_yaxes(title_text="MAE (log)", row=1, col=2)
    fig_curves.update_layout(
        title="Learning curves",
        height=380,
        margin=dict(l=10, r=10, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    )
    fig_curves
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parity on the held-out test set

    The val curve above tells us the model isn't overfitting; the parity plot
    below tells us *where* the remaining error is. We run the trained model
    on every test frame, recover absolute energies by adding the per-element
    shift back, and scatter predicted vs. true. A perfectly fit model lies on
    the dashed identity line.

    We do this manually rather than calling `evaluate(...)` so we can keep the
    raw arrays for plotting (MAE alone hides the shape of the errors).
    """)
    return


@app.cell
def _(
    apply_atomic_reference,
    compute_forces,
    mo,
    model,
    np,
    shifts,
    test_loader,
    torch,
):
    # Manual eval loop: keep the raw per-frame arrays for the parity plot.
    model.eval()
    _e_pred_all, _e_true_all = [], []
    _f_pred_all, _f_true_all = [], []

    for _batch in test_loader:
        _g = _batch["graph"]
        _g.pos.requires_grad_(True)
        # Predicted (residual) energy → add the reference back to recover absolute.
        _pred_residual = model(_g)  # [B]
        _pred_forces = compute_forces(_pred_residual.sum(), _g.pos)  # [N_total, 3]
        _ref = apply_atomic_reference(_g.z, _g.batch, shifts).to(_batch["energy"].dtype)
        _pred_abs = (_pred_residual + _ref).detach()

        _e_pred_all.append(_pred_abs)
        _e_true_all.append(_batch["energy"])
        _f_pred_all.append(_pred_forces.detach())
        _f_true_all.append(_batch["forces"])

    e_pred = torch.cat(_e_pred_all).numpy()
    e_true = torch.cat(_e_true_all).numpy()
    f_pred = torch.cat(_f_pred_all).numpy().reshape(-1)
    f_true = torch.cat(_f_true_all).numpy().reshape(-1)

    test_energy_mae = float(np.abs(e_pred - e_true).mean()) / 9  # 9 atoms per ethanol
    test_force_mae = float(np.abs(f_pred - f_true).mean())

    mo.md(
        f"**Test set ({len(e_true)} frames):** "
        f"per-atom `energy_mae = {test_energy_mae:.4f}` kcal/mol/atom &nbsp;·&nbsp; "
        f"`force_mae = {test_force_mae:.4f}` kcal/mol/Å"
    )
    return e_pred, e_true, f_pred, f_true, test_energy_mae, test_force_mae


@app.cell(hide_code=True)
def _(
    e_pred,
    e_true,
    f_pred,
    f_true,
    go,
    make_subplots,
    np,
    test_energy_mae,
    test_force_mae,
):
    fig_parity = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Energy parity (per-atom MAE = {test_energy_mae:.4f} kcal/mol)",
            f"Force component parity (MAE = {test_force_mae:.4f} kcal/mol/Å)",
        ),
        horizontal_spacing=0.12,
    )

    _e_lo, _e_hi = float(min(e_true.min(), e_pred.min())), float(max(e_true.max(), e_pred.max()))
    fig_parity.add_trace(
        go.Scatter(
            x=e_true,
            y=e_pred,
            mode="markers",
            marker=dict(color="#1f77b4", size=6, opacity=0.7),
            showlegend=False,
            hovertemplate="true=%{x:.2f}<br>pred=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_parity.add_trace(
        go.Scatter(
            x=[_e_lo, _e_hi],
            y=[_e_lo, _e_hi],
            mode="lines",
            line=dict(dash="dash", color="#888"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    _f_lo, _f_hi = float(min(f_true.min(), f_pred.min())), float(max(f_true.max(), f_pred.max()))
    # Subsample to keep the scatter trace light (2700 points hovered is fine,
    # but 100k+ atoms x 3 components on a larger molecule would not be).
    _rng = np.random.default_rng(0)
    _idx = _rng.choice(len(f_true), size=min(1500, len(f_true)), replace=False)
    fig_parity.add_trace(
        go.Scatter(
            x=f_true[_idx],
            y=f_pred[_idx],
            mode="markers",
            marker=dict(color="#ff7f0e", size=4, opacity=0.5),
            showlegend=False,
            hovertemplate="true=%{x:.2f}<br>pred=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig_parity.add_trace(
        go.Scatter(
            x=[_f_lo, _f_hi],
            y=[_f_lo, _f_hi],
            mode="lines",
            line=dict(dash="dash", color="#888"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig_parity.update_xaxes(title_text="true energy (kcal/mol)", row=1, col=1)
    fig_parity.update_yaxes(title_text="predicted energy (kcal/mol)", row=1, col=1)
    fig_parity.update_xaxes(title_text="true F component (kcal/mol/Å)", row=1, col=2)
    fig_parity.update_yaxes(title_text="predicted F component (kcal/mol/Å)", row=1, col=2)
    fig_parity.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=40))
    fig_parity
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Now look at the forces

    Remember the random arrows in notebook 03? Same plot, same molecule, but
    the model is trained now. We pick one test frame, compute autograd forces
    through the trained model, and overlay them on the *true* rMD17 forces:

    - **crimson** arrows: predicted forces from the trained MLIP
    - **royal blue** arrows: rMD17 reference forces from the DFT calculation

    Two cones nearly on top of each other at every atom means the model has
    learned the local chemistry well enough for MD. A frame index slider lets
    you scrub through the test set.
    """)
    return


@app.cell(hide_code=True)
def _(mo, test_bundle):
    frame_idx = mo.ui.slider(
        start=0,
        stop=len(test_bundle.structures) - 1,
        step=1,
        value=0,
        label="test frame",
        show_value=True,
    )
    arrow_scale = mo.ui.slider(
        start=0.01,
        stop=0.5,
        step=0.01,
        value=0.1,
        label="arrow scale",
        show_value=True,
    )
    mo.vstack([frame_idx, arrow_scale])
    return arrow_scale, frame_idx


@app.cell(hide_code=True)
def _(
    arrow_scale,
    build_graph,
    compute_forces,
    frame_idx,
    go,
    model,
    np,
    test_bundle,
    tiny,
):
    from ase.data import chemical_symbols as _chem

    from tinymlip.viz import element_color, element_radius

    _atoms = test_bundle.structures[frame_idx.value]
    _graph_arrow = build_graph(_atoms, cutoff=tiny["cutoff"])
    _graph_arrow.pos.requires_grad_(True)
    _pred_res = model(_graph_arrow)
    _pred_force_one = compute_forces(_pred_res, _graph_arrow.pos).detach().numpy()
    _true_force_one = _atoms.arrays["forces"]
    _pos_np = _graph_arrow.pos.detach().numpy()
    _z_np = _graph_arrow.z.numpy()

    # Bonds: tighter cutoff just for drawing the molecular skeleton.
    _graph_bonds = build_graph(_atoms, cutoff=1.6)
    _bs, _bd = _graph_bonds.edge_index
    _bond_x, _bond_y, _bond_z = [], [], []
    for _s, _d in zip(_bs.tolist(), _bd.tolist(), strict=True):
        if _s < _d:
            _bond_x.extend([_pos_np[_s, 0], _pos_np[_d, 0], None])
            _bond_y.extend([_pos_np[_s, 1], _pos_np[_d, 1], None])
            _bond_z.extend([_pos_np[_s, 2], _pos_np[_d, 2], None])

    _scale = float(arrow_scale.value)

    def _arrow_traces(forces, color, name):
        tips = _pos_np + _scale * forces
        sx, sy, sz = [], [], []
        for k in range(_graph_arrow.n_atoms):
            sx.extend([_pos_np[k, 0], tips[k, 0], None])
            sy.extend([_pos_np[k, 1], tips[k, 1], None])
            sz.extend([_pos_np[k, 2], tips[k, 2], None])
        return [
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="lines",
                line=dict(color=color, width=4),
                hoverinfo="skip",
                showlegend=True,
                name=name,
            ),
            go.Cone(
                x=tips[:, 0],
                y=tips[:, 1],
                z=tips[:, 2],
                u=_scale * forces[:, 0],
                v=_scale * forces[:, 1],
                w=_scale * forces[:, 2],
                anchor="tail",
                sizemode="absolute",
                sizeref=3.0 * _scale,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                hoverinfo="skip",
                showlegend=False,
            ),
        ]

    fig_arrows = go.Figure()
    fig_arrows.add_trace(
        go.Scatter3d(
            x=_bond_x,
            y=_bond_y,
            z=_bond_z,
            mode="lines",
            line=dict(color="#bbbbbb", width=3),
            hoverinfo="skip",
            showlegend=False,
            name="bonds",
        )
    )
    for trace in _arrow_traces(_true_force_one, "royalblue", "true (rMD17 DFT)"):
        fig_arrows.add_trace(trace)
    for trace in _arrow_traces(_pred_force_one, "crimson", "predicted (MLIP)"):
        fig_arrows.add_trace(trace)
    fig_arrows.add_trace(
        go.Scatter3d(
            x=_pos_np[:, 0],
            y=_pos_np[:, 1],
            z=_pos_np[:, 2],
            mode="markers+text",
            marker=dict(
                size=[element_radius(int(z)) * 14 for z in _z_np],
                color=[element_color(int(z)) for z in _z_np],
                line=dict(color="#222", width=1),
            ),
            text=[f"{_chem[int(z)]}[{k}]" for k, z in enumerate(_z_np)],
            textposition="top center",
            textfont=dict(size=10, color="#111"),
            hoverinfo="skip",
            showlegend=False,
            name="atoms",
        )
    )
    _pred_fmag = np.linalg.norm(_pred_force_one, axis=-1).max()
    _true_fmag = np.linalg.norm(_true_force_one, axis=-1).max()
    fig_arrows.update_layout(
        title=(
            f"Frame {frame_idx.value}: |F_pred|_max = {_pred_fmag:.2f}, "
            f"|F_true|_max = {_true_fmag:.2f} (kcal/mol/Å), "
            f"arrows × {_scale:.2f}"
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            dragmode="turntable",
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
    )
    fig_arrows
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we deferred (and where it goes)

    The training loop above is intentionally minimal. Three things that are
    NOT here:

    - **Equivariant message passing.** Our scalar messages discard the
      *direction* an edge points. Notebook 05 trains `EquivariantMPNN` on the
      exact same loop and compares the two side-by-side.
    - **Periodic systems.** rMD17 is gas-phase molecules. Notebook 06 adds
      PBC support to the neighbor list and demos the model as an ASE
      calculator on a small crystal.
    - **Production tricks.** LR schedulers, EMA, gradient clipping, mixed
      precision, early stopping — all routinely used by SchNetPack, NequIP,
      MACE. They're outside the scope of this notebook, but adding them on top
      of `train` is straightforward.
    """)
    return


if __name__ == "__main__":
    app.run()
