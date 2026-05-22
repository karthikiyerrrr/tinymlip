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
    import ase.build
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import torch
    from ase.calculators.emt import EMT
    from torch.utils.data import DataLoader

    from tinymlip.data import (
        load_cu_emt,
        make_collate,
        to_torch_dataset_cu_emt,
    )
    from tinymlip.forces import compute_forces_and_stress
    from tinymlip.graph import build_graph
    from tinymlip.models import EquivariantMPNN
    from tinymlip.train import evaluate, fit_atomic_reference, train
    from tinymlip.viz import e_v_curve

    return (
        DataLoader,
        EMT,
        EquivariantMPNN,
        ase,
        build_graph,
        compute_forces_and_stress,
        e_v_curve,
        evaluate,
        fit_atomic_reference,
        go,
        load_cu_emt,
        make_collate,
        np,
        pl,
        to_torch_dataset_cu_emt,
        torch,
        train,
    )


@app.cell(hide_code=True)
def _(mo):
    a_slider = mo.ui.slider(
        start=3.0, stop=4.5, step=0.05, value=3.615, label="lattice constant a (Å)"
    )
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

    **What the plot does and doesn't show.** The visualization filters to
    edges with `S_z = 0` for clarity; the full PBC graph also has edges
    connecting through the top and bottom faces. An atom can also interact
    with *its own image* in an adjacent cell — `S = (1, 0, 0)` on a boundary
    atom produces an edge from the atom to itself, displaced by `+a₁`. This
    is a real PBC effect, not a bug.

    **Cutoff vs. cell size (minimum-image convention).** MD codes typically
    enforce `cutoff < L/2` so an atom can't feel multiple images of the same
    physical neighbor. Our single-cell `a ≈ 3.6 Å` puts `L/2 ≈ 1.8 Å` against
    the default cutoff `4.0 Å`, so we are *deliberately above* that limit —
    the neighbor list still handles it correctly per-edge via `shift_idx`
    (each image is a distinct edge), but in production you'd use a larger
    supercell or a tighter cutoff.
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
                corners_x = [
                    origin[0],
                    origin[0] + a1[0],
                    origin[0] + a1[0] + a2[0],
                    origin[0] + a2[0],
                    origin[0],
                ]
                corners_y = [
                    origin[1],
                    origin[1] + a1[1],
                    origin[1] + a1[1] + a2[1],
                    origin[1] + a2[1],
                    origin[1],
                ]
                is_central = i == 0 and j == 0
                fig.add_trace(
                    go.Scatter(
                        x=corners_x,
                        y=corners_y,
                        mode="lines",
                        line=dict(
                            color="#444" if is_central else "#bbb", width=2 if is_central else 1
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                # Atoms in this tile
                xs = pos[:, 0] + origin[0]
                ys = pos[:, 1] + origin[1]
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        marker=dict(
                            size=14 if is_central else 9,
                            color="#b87333" if is_central else "#e7c9a8",
                            line=dict(color="#333", width=1),
                        ),
                        hoverinfo="text",
                        text=[f"atom {k} tile ({i},{j})" for k in range(len(pos))],
                        showlegend=False,
                    )
                )

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
            fig.add_trace(
                go.Scatter(
                    x=[a[0], b[0]],
                    y=[a[1], b[1]],
                    mode="lines",
                    line=dict(color=color, width=1.2),
                    name=name,
                    legendgroup=name,
                    showlegend=show,
                    hoverinfo="skip",
                )
            )

        a = atoms.cell.array[0, 0]
        fig.update_layout(
            xaxis=dict(title="x (Å)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="y (Å)"),
            title=f"FCC Cu (a={a:.2f} Å), central cell + 8 image cells (z=0 slice)",
            width=600,
            height=600,
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
    This is the **thermodynamic definition of Cauchy stress**:
    `σ_ij = ∂W/∂ε_ij` with strain energy density `W = E/V`. Stress is the
    conjugate variable to strain in the elastic free energy, the same way
    force is the conjugate to position. It is *not* a trick — it is the
    definition you'd find in any continuum-mechanics text. What is new here
    is only that we can take that derivative *by autograd* against the
    learned energy.

    Parameterize the deformation as a **symmetric** 3×3 strain tensor `ε`.
    Atomic positions deform as `r → r·(I + ε)` and the cell as
    `c → c·(I + ε)`. The determinant of `I + ε` is `1 + tr(ε) + O(ε²)`, so a
    uniform isotropic strain is just a volumetric scaling — exactly what the
    `a` slider above does, in disguise.

    **Why ε is symmetric.** The antisymmetric part of an infinitesimal
    deformation is an infinitesimal rotation — energy is rotation-invariant
    (nb05's whole point), so the antisymmetric channel can't carry any
    signal. Symmetrizing ε projects out a gauge degree of freedom rather
    than imposing a physical constraint. (Engineering codes ship stress as
    the Voigt 6-vector `(σ_11, σ_22, σ_33, σ_23, σ_13, σ_12)`; ASE's
    `get_stress(voigt=False)` is named *false* because it returns the full
    3×3 we use here.)

    **Stress is then `σ = (1/V) · ∂E/∂ε`,** evaluated at `ε = 0`. If `ε` is
    a leaf in the autograd graph, σ falls out of one extra backward pass —
    the same trick that gave us forces from `−∂E/∂r` back in nb03. We get a
    rank-2 tensor instead of a vector, but the principle is identical.

    **Sign convention.** `F = −∂E/∂r` carries a minus (force points downhill
    in energy); `σ = +(1/V) ∂E/∂ε` does not. This is not an inconsistency —
    it is the standard tension-positive Cauchy convention: positive `σ_xx`
    means the lattice is under tensile load in `x`. ASE uses the same
    convention, which is why predicted and reference σ line up later
    without any sign flip.

    **One scalar, three observables, one rule.** Energy is the leaf of the
    model; every physical quantity is a derivative of `E` against the right
    upstream leaf. nb03's rule was: `pos` must be the leaf, not a cached
    `edge_vec`, so `∂E/∂pos` actually flows. nb06's rule is the same in
    different coin: the **cell tensor** must be in the autograd graph, not
    bypassed by a cached `edge_vec`, so `∂E/∂cell` actually flows. Same
    trick, different leaf. Our `InvariantInteraction` and
    `EquivariantInteraction` both recompute
    `edge_vec = pos[j] − pos[i] + S · cell` *inside* the forward pass for
    exactly this reason; the next cell verifies σ against a hand-rolled
    numerical derivative to confirm.

    <details>
    <summary><b>Try this:</b> what does σ come out as if you skip the
    <code>pos + S @ cell</code> recompute inside
    <code>EquivariantInteraction.forward</code>?</summary>

    σ collapses to ~0 to machine precision, because the cell tensor never
    enters the autograd graph and `∂E/∂ε` is identically zero at the leaf.
    Concretely: comment out the `if graph.shift_idx is not None: ...` re-add
    block in `layers.EquivariantInteraction.forward` (and the matching one
    in `InvariantInteraction.forward`), rerun the parity cell below, and
    the autograd σ row in the table reads all zeros. The numerical σ row
    is unchanged — finite-differencing the forward pass doesn't need the
    cell to be a leaf.
    </details>
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
            ep = torch.zeros(3, 3, dtype=torch.float64)
            ep[i, j] += h
            ep[j, i] += h
            em = torch.zeros(3, 3, dtype=torch.float64)
            em[i, j] -= h
            em[j, i] -= h
            e_plus = demo_model(
                _replace(
                    demo_g,
                    pos=demo_g.pos + demo_g.pos @ ep,
                    cell=demo_g.cell + demo_g.cell @ ep,
                )
            ).sum()
            e_minus = demo_model(
                _replace(
                    demo_g,
                    pos=demo_g.pos + demo_g.pos @ em,
                    cell=demo_g.cell + demo_g.cell @ em,
                )
            ).sum()
            sigma_num[i, j] = (e_plus - e_minus) / (4 * h) / volume
    sigma_num = 0.5 * (sigma_num + sigma_num.T)

    max_abs_err = (sigma_ad - sigma_num).abs().max().item()
    _s_ad = sigma_ad.detach().numpy()
    _s_nm = sigma_num.detach().numpy()

    mo.vstack(
        [
            mo.md(f"**max |σ_autograd − σ_numerical| = {max_abs_err:.3e}**  (target: < 1e-5)"),
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
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Parity confirmed.** Autograd σ matches a central finite-difference of
    the energy under a symmetrized hand-built strain to roughly 1e-8. This
    is the same principle as the force parity check in nb03 — energy is the
    single learned scalar, every other physical quantity is a derivative of
    it against a different leaf (positions → forces, cell strain → stress).

    The strain trick is numerically *exact*: this fp64 parity confirms the
    autograd derivation to ~1e-8. The fp32 arithmetic of the trained model
    in later sections loses some precision in its own internal ops, but
    that is unrelated — the `σ = (1/V) ∂E/∂ε` identity itself stays
    bit-accurate at whatever precision the model runs in.

    We use a small `EquivariantMPNN(n_layers=2, hidden_dim=16)` here just to
    make the parity check fast; the trained model later uses a larger
    configuration. Parity is about the autograd machinery, not the trained
    weights — random weights work fine, and smaller is faster.

    The diagonal entries above are the **normal stresses** (eV/Å³); the
    off-diagonals are **shears**. On a perfectly equilibrated lattice all
    six independent entries would be zero, but our untrained model has
    random weights and the lattice is slightly rattled, so non-zero values
    are expected. After training the entries should track ASE's EMT σ to
    within the test-set MAE.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the labels come from

    We need labeled data to train against. **ASE ships an EMT
    (Effective Medium Theory) calculator** that works on FCC metals
    — Cu, Al, Ni, Pd, Ag, Pt, Au — and produces energy, forces, **and**
    stress for free in physical units (eV, eV/Å, eV/Å³).

    EMT is an embedded-atom-method-style classical potential — each atom's
    energy depends on the *electron density* it sees from its neighbors.
    Jacobsen, Stoltze, and Nørskov (1996) parameterized it against DFT
    cohesive properties of those seven FCC metals: it captures the right
    elastic constants, lattice constants, and vacancy energies to within
    ~10% of DFT, fast enough that ASE ships it as a built-in. We use it
    because it's calibrated and fast, not because it's mysterious. In
    production you'd swap EMT for GAP / DFT / OC20 labels — and nothing
    else in this notebook would change.

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

    mo.vstack(
        [
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
        ]
    )
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

    mo.vstack(
        [
            mo.md(f"Generated/loaded **{len(all_atoms)}** snapshots in **{elapsed:.1f} s**"),
            meta.head(),
        ]
    )
    return all_atoms, meta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the equivariant model on (E, F, σ)

    We minimize a weighted sum of mean-absolute errors:

    \[
        \mathcal{L} = w_E \cdot \mathrm{MAE}(E) + w_F \cdot \mathrm{MAE}(F) + w_\sigma \cdot \mathrm{MAE}(\sigma).
    \]

    The defaults at `tiny` are `w_E = 1`, `w_F = 100`, `w_σ = 10`:

    - `w_E = 1` — energies are ~10 eV magnitude per snapshot; MAE in the
      ones-of-meV range, so a coefficient of 1 puts the energy term at the
      same scale as the others below.
    - `w_F = 100` — force components are ~eV/Å magnitude; an MAE of 10 meV/Å
      is excellent, and a unit weight would let the energy MAE dominate.
      The factor of 100 brings the force MAE up to a comparable contribution.
    - `w_σ = 10` — stress components are ~meV/Å³ magnitude (much smaller
      than forces in absolute units); a coefficient of 10 is enough to make
      the stress term visible but not dominant. Setting `w_σ = 0` recovers
      the nb05 loss exactly.

    `fit_atomic_reference` removes the per-element baseline so the model
    only has to learn the *deviation* from the average per-atom energy — a
    free win that we've used since nb04.

    **Knob inventory.** The slider row exposes `w_σ`, `n_epochs`, and
    `hidden_dim`. `n_layers = 3`, `num_basis = 20`, and `cutoff = 4.0 Å`
    are pinned to values that fit the 5-minute CPU budget for `tiny`; the
    `small` / `default` configs in `configs/` widen them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    w_s_slider = mo.ui.slider(start=0.0, stop=100.0, step=1.0, value=10.0, label="w_σ")
    n_epochs_slider = mo.ui.slider(start=5, stop=60, step=5, value=30, label="n_epochs")
    hidden_slider = mo.ui.slider(start=8, stop=64, step=8, value=32, label="hidden_dim")
    mo.hstack([w_s_slider, n_epochs_slider, hidden_slider])
    return hidden_slider, n_epochs_slider, w_s_slider


@app.cell
def _(
    DataLoader,
    EquivariantMPNN,
    all_atoms,
    fit_atomic_reference,
    hidden_slider,
    make_collate,
    meta,
    mo,
    n_epochs_slider,
    np,
    to_torch_dataset_cu_emt,
    torch,
    train,
    w_s_slider,
):
    train_atoms = [a for a, s in zip(all_atoms, meta["split"], strict=True) if s == "train"]
    val_atoms = [a for a, s in zip(all_atoms, meta["split"], strict=True) if s == "val"]
    test_atoms = [a for a, s in zip(all_atoms, meta["split"], strict=True) if s == "test"]

    shifts = fit_atomic_reference(
        train_atoms,
        np.array([a.info["energy"] for a in train_atoms]),
    )

    collate = make_collate(cutoff=4.0)
    train_loader = DataLoader(
        to_torch_dataset_cu_emt(train_atoms),
        batch_size=8,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        to_torch_dataset_cu_emt(val_atoms),
        batch_size=16,
        shuffle=False,
        collate_fn=collate,
    )

    torch.manual_seed(0)
    model = EquivariantMPNN(
        n_layers=3,
        hidden_dim=hidden_slider.value,
        num_basis=20,
        cutoff=4.0,
    )
    log = train(
        model,
        train_loader,
        val_loader,
        n_epochs=n_epochs_slider.value,
        lr=1e-3,
        w_e=1.0,
        w_f=100.0,
        w_s=w_s_slider.value,
        shifts=shifts,
    )

    mo.vstack(
        [
            mo.md(
                f"""
    - **Splits:** {len(train_atoms)} train / {len(val_atoms)} val / {len(test_atoms)} test
    - **Per-Cu reference shift:** {shifts[29]:.4f} eV
    - **Training:** EquivariantMPNN(n_layers=3, hidden_dim={hidden_slider.value}, num_basis=20, cutoff=4.0) for {n_epochs_slider.value} epochs
    """
            ),
            mo.md("**Last 6 log rows:**"),
            log.tail(6),
        ]
    )
    return collate, log, model, shifts, test_atoms


@app.cell(hide_code=True)
def _(go, log, pl):
    _fig = go.Figure()
    for _split, _color in [("train", "steelblue"), ("val", "indianred")]:
        _sub = log.filter(pl.col("split") == _split)
        _fig.add_trace(
            go.Scatter(
                x=_sub["epoch"],
                y=_sub["loss"],
                mode="lines+markers",
                name=f"{_split} loss",
                line=dict(color=_color),
            )
        )
    _fig.update_layout(
        xaxis_title="epoch",
        yaxis_title="loss",
        yaxis_type="log",
        title="Training curve — Cu/EMT, E + F + σ loss",
        width=700,
        height=400,
    )
    _fig
    return


@app.cell
def _(
    DataLoader,
    collate,
    evaluate,
    mo,
    model,
    pl,
    shifts,
    test_atoms,
    to_torch_dataset_cu_emt,
    w_s_slider,
):
    test_loader = DataLoader(
        to_torch_dataset_cu_emt(test_atoms),
        batch_size=16,
        shuffle=False,
        collate_fn=collate,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        shifts=shifts,
        w_e=1.0,
        w_f=100.0,
        w_s=w_s_slider.value,
    )
    _n_atoms_per_snap = test_atoms[0].get_global_number_of_atoms()

    mo.vstack(
        [
            mo.md(f"**Test-set MAEs** ({len(test_atoms)} snapshots, batch 16):"),
            pl.DataFrame(
                {
                    "metric": [
                        "energy (meV / atom)",
                        "force component (meV / Å)",
                        "stress component (meV / Å³)",
                        "total loss",
                    ],
                    "value": [
                        f"{1000 * test_metrics['energy_mae'] / _n_atoms_per_snap:.3f}",
                        f"{1000 * test_metrics['force_mae']:.3f}",
                        f"{1000 * test_metrics['stress_mae']:.3f}"
                        if "stress_mae" in test_metrics
                        else "—",
                        f"{test_metrics['loss']:.4f}",
                    ],
                }
            ),
            mo.md(
                r"""
    **Are these numbers good?** On FCC-metal MLIPs trained at scale, ~1 meV/Å
    force MAE is near-SOTA; on this small synthetic dataset, 5–20 meV/Å is
    typical and the DFT–DFT spread across functionals is itself ~5 meV/Å. For
    stress, the conversion `1 eV/Å³ ≈ 160 GPa` makes `5 meV/Å³` MAE ≈ 1 GPa —
    in the same ballpark as the DFT-DFT functional spread. A few meV/atom on
    energy is the standard ambition; 1 meV/atom is the target on real
    materials datasets.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verification: does the trained model respect the physics?

    Three sanity checks on the **test set** (snapshots the model never saw):

    1. **Extensivity** — replicate a snapshot, energy should double and
       stress should stay the same.
    2. **E–V curve** — predicted E(V) under isotropic volumetric strain
       should track ASE's EMT reference.
    3. **Rotation equivariance** — under a random proper rotation `R`,
       energy should be invariant, forces should rotate as `F → F·Rᵀ`, and
       stress should rotate as `σ → R σ Rᵀ`.

    fp32 has roughly seven digits of precision, so a 1e-4 max-error on the
    rotation identities below is at the noise floor for a model with
    O(10⁴) accumulated ops per atom — same identities, just measured
    against the fp32 floor instead of fp64.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Supercell extensivity

    Replicating a snapshot 2× along one axis doubles every atom; an
    extensive quantity (energy) should double, an intensive one (stress)
    should not change.
    """)
    return


@app.cell
def _(
    build_graph,
    compute_forces_and_stress,
    mo,
    model,
    pl,
    test_atoms,
    torch,
):
    _test_atom_0 = test_atoms[0]
    _big = _test_atom_0.repeat((2, 1, 1))

    _g_small = build_graph(_test_atom_0, cutoff=4.0, dtype=torch.float32)
    _g_big = build_graph(_big, cutoff=4.0, dtype=torch.float32)
    _E_s, _, _sigma_s = compute_forces_and_stress(model, _g_small)
    _E_b, _, _sigma_b = compute_forces_and_stress(model, _g_big)

    _ratio = (_E_b / _E_s).item()
    _sigma_diff = (_sigma_b - _sigma_s).abs().max().item()

    mo.vstack(
        [
            pl.DataFrame(
                {
                    "quantity": [
                        "E_small (eV)",
                        "E_big (eV)",
                        "E_big / E_small",
                        "max |σ_big − σ_small| (eV/Å³)",
                    ],
                    "value": [
                        f"{_E_s.item():.4f}",
                        f"{_E_b.item():.4f}",
                        f"{_ratio:.4f}  (expect ≈ 2.0)",
                        f"{_sigma_diff:.3e}  (expect ≈ 0)",
                    ],
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Energy–volume curve vs EMT

    For each volume fraction `f ∈ [0.9, 1.1]`, isotropically scale the
    base snapshot's cell and atoms by `f^(1/3)`, run both the model and
    EMT, and overlay the resulting `E(V)` curves. The two curves should
    track each other; deviations are the model's generalization error to
    deformations it didn't see during training.
    """)
    return


@app.cell
def _(EMT, e_v_curve, model, np, shifts, test_atoms):
    _base = test_atoms[0]
    _volume_fractions = np.linspace(0.9, 1.1, 11)

    _ref_energies = []
    for _f in _volume_fractions:
        _a = _base.copy()
        _a.set_cell(_base.cell.array * float(_f) ** (1 / 3), scale_atoms=True)
        _a.calc = EMT()
        _ref_energies.append(_a.get_potential_energy())

    _ev_fig = e_v_curve(
        model,
        _base,
        list(_volume_fractions),
        reference_energies=_ref_energies,
        shifts=shifts,
    )
    _ev_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for here.** Both curves are now in physical eV units
    (the model's reference-shift `Σ shifts[z]` is added back per atom
    inside `e_v_curve`), so the absolute offset is a real comparison.

    Two things you should see, and what they tell you:

    1. **Both wells are parabolic-ish** with a clear minimum. Good — the
       model has learned the qualitative cohesive behavior of FCC Cu
       (compression hurts, expansion hurts, equilibrium in between).
    2. **The model's minimum may sit at slightly larger volume than
       EMT's**, and the model's curvature may be softer than EMT's. This
       is a real generalization error: our training data was concentrated
       near equilibrium (`strain_range=0.05`, `rattle_amp=0.1`), so the
       model has limited signal on the deeper-compression tail where the
       curves diverge most. Longer training, wider strain sampling, or a
       stronger energy weight would tighten the well.

    **What this curve encodes.** A materials scientist looking at an E(V)
    plot immediately wants `B = V · ∂²E/∂V²` at the minimum — the **bulk
    modulus**, a fundamental measurable property of any solid. EMT Cu's
    B ≈ 134 GPa; DFT Cu ≈ 140 GPa; experimental ≈ 137 GPa. The trained
    MLIP gives you the full anisotropic elastic-constant tensor `C_ij` for
    free: they are second derivatives of `E` with respect to strain, and
    we already wired up the first derivative (σ). One more backward pass
    (or finite-differencing σ vs. ε) yields the full `C_ij` — for cubic
    FCC Cu, `C_11 ≈ 175 GPa`, `C_12 ≈ 130 GPa`, `C_44 ≈ 82 GPa`. *E(V) →
    B; σ vs. ε → C_ij.* The parity-check plot is also a property-
    prediction plot.

    For a teaching demo, this is exactly the lesson: a trained MLIP
    interpolates well in the regime it saw, and extrapolates with
    increasing error outside it. A bigger dataset and more epochs would
    close the well-shape gap; a fundamentally bigger gap would point at a
    capacity or featurization problem instead.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Rotation equivariance of energy, forces, and stress

    Apply a random proper rotation `R ∈ SO(3)` to both the cell and the
    atomic positions of the same snapshot. The graph builder produces a
    new (but physically equivalent) graph; the equivariant model should
    map outputs the standard way: scalars invariant, vectors and rank-2
    tensors rotated.
    """)
    return


@app.cell
def _(build_graph, compute_forces_and_stress, model, pl, test_atoms, torch):
    torch.manual_seed(7)
    _A = torch.randn(3, 3)
    _Q, _ = torch.linalg.qr(_A)
    if torch.det(_Q) < 0:
        _Q[:, 0] *= -1
    _R_np = _Q.numpy()

    _rot = test_atoms[0].copy()
    _rot.set_cell(test_atoms[0].cell.array @ _R_np.T, scale_atoms=False)
    _rot.set_positions(test_atoms[0].positions @ _R_np.T)

    _g_orig = build_graph(test_atoms[0], cutoff=4.0, dtype=torch.float32)
    _g_rot = build_graph(_rot, cutoff=4.0, dtype=torch.float32)
    _E_a, _F_a, _s_a = compute_forces_and_stress(model, _g_orig)
    _E_b, _F_b, _s_b = compute_forces_and_stress(model, _g_rot)

    _R_t = torch.as_tensor(_R_np, dtype=_F_a.dtype)
    _e_err = (_E_b - _E_a).abs().item()
    _f_err = (_F_b - _F_a @ _R_t.T).abs().max().item()
    _s_err = (_s_b - _R_t @ _s_a @ _R_t.T).abs().max().item()

    pl.DataFrame(
        {
            "identity": [
                "scalar:  E(rot) = E",
                "vector:  F(rot) = F · Rᵀ",
                "tensor:  σ(rot) = R · σ · Rᵀ",
            ],
            "max abs error": [
                f"{_e_err:.3e}",
                f"{_f_err:.3e}",
                f"{_s_err:.3e}",
            ],
            "interpretation": [
                "expect ≈ 0  (energy is rotation-invariant)",
                "expect ≈ 0  (forces transform as vectors)",
                "expect ≈ 0  (stress transforms as a rank-2 tensor)",
            ],
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we built, and where to go from here

    **The MLIP framework, in one sentence.** A neural network learns one
    scalar function `E(positions, cell, species)`. Every physical quantity
    is a derivative of it against the right leaf:

    - position → forces (nb03)
    - strain → stress (nb06)
    - composition → chemical potential
    - electric field → dipole / polarizability

    Every "new" quantity is a new autograd call, not a new model. The
    conservative-by-construction property (no separate force head, no
    separate stress head) is what makes MLIPs work in molecular dynamics:
    the trained potential exactly conserves energy in NVE, exactly
    satisfies the virial theorem in NVT, and exactly respects the
    stress–strain identity in NPT — because all those identities follow
    from a *single* `E` being the source of every observable.

    **What we did.** Took the equivariant message-passing model from nb05
    to periodic crystals. We:

    - Added integer lattice shifts `S` to every edge so graphs can cross
      the unit-cell boundary.
    - Recomputed `edge_vec = pos[j] − pos[i] + S · cell` *inside* each
      interaction layer so the cell tensor is part of the autograd graph.
    - Derived stress from the same learned energy via the strain trick
      `σ = (1/V) ∂E/∂ε` — one extra backward pass on a different leaf.
    - Trained on 800 synthetic FCC-Cu snapshots labeled by ASE's EMT,
      combining per-atom energy MAE, force MAE, and stress MAE in the
      loss. The training / loss / eval code is **the same code path**
      nb05 used — just with `w_σ > 0`. Setting `w_σ = 0` recovers nb05's
      loss exactly: one training loop, every observable, one weight knob.

    **Architecture vs. dataset.** The architecture has **no FCC assumption
    baked in.** The same `EquivariantMPNN` + PBC machinery would train on
    rattled BCC Fe, diamond Si, rutile TiO₂, or a Pt/MOF interface — swap
    the `Atoms` generator and the labeler; nothing in `models.py` /
    `layers.py` / `forces.py` changes. What this *dataset* doesn't cover
    and would therefore predict badly on: vacancies, surfaces, grain
    boundaries, phase transitions (FCC↔HCP), other compositions. Our
    32-atom rattled supercell covers the harmonic well around equilibrium
    FCC and nothing else. **MLIP quality is dataset-bounded, not
    architecture-bounded.**

    **What we deliberately skipped.**

    - **Long-range Coulomb / Ewald sums.** Cu is metallic; EMT is
      short-ranged. Ionic crystals like NaCl would need Ewald.
    - **Real DFT labels.** We used a classical many-body potential as
      ground-truth; in real research labels come from DFT (MP-traj,
      GAP-18 Si, OC20 catalysis, …).
    - **NPT molecular dynamics with the trained potential.** The trained
      model is a drop-in ASE calculator candidate; running a barostat
      with it is a natural next step but out of scope here.
    - **Multi-species crystals.** Single-element Cu keeps the data layer
      identical to nb04/nb05.

    **Where to go from here.** Production equivariant MLIPs use the
    [e3nn](https://e3nn.org) library for higher-order tensor
    representations (NequIP, MACE, Allegro). Datasets matched to nb06's
    PBC framing: **MPtraj** and **OC20** (both crystal-scale). Datasets
    matched to nb05's molecular framing: **ANI-1x** and **SPICE**.
    e3nn-based NequIP / MACE / Allegro work in both regimes. For end-to-
    end demos including NPT MD, see the ASE + MACE tutorials.
    """)
    return


if __name__ == "__main__":
    app.run()
