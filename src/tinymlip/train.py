"""Training utilities for tinymlip's MPNN models.

Composable functions used by notebook 04 (invariant) and 05 (equivariant):

  - fit_atomic_reference / apply_atomic_reference: per-element energy offsets.
    Subtracted from targets before training so the model only learns small
    residuals (~kcal/mol) instead of absolute energies (~-4000 kcal/mol).
    Mirrors SchNetPack's atomref machinery.
  - energy_force_loss: weighted per-atom energy MSE + force MSE, with MAE
    metrics returned alongside for human-readable logging.
  - train_one_epoch / evaluate / train: the loop itself. Returns a polars
    DataFrame run log so the notebook can plot directly.

References:
  - SchNetPack `estimate_atomrefs` (src/schnetpack/data/stats.py): the same
    least-squares fit, slightly different storage (tensor of length z_max
    vs our dict[int, float] — we prefer the dict for readability).
  - SchNetPack `RemoveOffsets`/`AddOffsets` (src/schnetpack/transform/atomistic.py):
    same sign convention — subtract on the way in, add on the way out.

Forces are NEVER predicted directly; they are autograd-derived through
`tinymlip.forces.compute_forces`. The training loop sets pos.requires_grad_(True)
before each forward so that path is live.
"""

from __future__ import annotations

import ase
import numpy as np
import polars as pl
import torch
from torch import Tensor

from tinymlip.forces import compute_forces, compute_forces_and_stress


def fit_atomic_reference(
    structures: list[ase.Atoms],
    energies: np.ndarray,
) -> dict[int, float]:
    """Fit per-element energy offsets via least squares: E_frame ≈ Σ_i shift[z_i].

    Args:
        structures: list of ASE Atoms, one per frame in the training set.
        energies:   [n_frames] array of reference energies (same units as the
                    targets the model will see — e.g. kcal/mol for rMD17).

    Returns:
        Mapping {atomic_number -> shift}. Subtract `sum(shift[z_i] for z_i in
        atoms.numbers)` from a frame's energy before training, add it back at
        inference. Elements not in the training set are NOT in the dict and
        will raise a KeyError if encountered later — that's intentional, since
        the fit has no information about them.
    """
    if len(structures) != len(energies):
        raise ValueError(f"structures and energies disagree: {len(structures)} vs {len(energies)}")
    # Discover which atomic numbers actually appear in the data — keeps the
    # composition matrix as small as possible and avoids a singular system if
    # an element is absent.
    elements = sorted({int(z) for atoms in structures for z in atoms.numbers})

    # design_matrix[i, j] = count of element elements[j] in structure i.
    # Then E ≈ design_matrix @ w, w ∈ R^|elements|. Solve via np.linalg.lstsq
    # (robust to rank deficiency, returns least-norm solution if the matrix is
    # rank-deficient).
    n_frames = len(structures)
    design_matrix = np.zeros((n_frames, len(elements)), dtype=np.float64)
    for i, atoms in enumerate(structures):
        for z in atoms.numbers:
            design_matrix[i, elements.index(int(z))] += 1.0
    w, *_ = np.linalg.lstsq(design_matrix, np.asarray(energies, dtype=np.float64), rcond=None)
    return {int(z): float(w[i]) for i, z in enumerate(elements)}


def apply_atomic_reference(
    z: Tensor,
    batch: Tensor | None,
    shifts: dict[int, float],
) -> Tensor:
    """Compute the per-frame atomic-reference offset Σ_i shift[z_i].

    This is the value you SUBTRACT from true energies before training, and ADD
    to predicted energies at inference time (to get back to absolute units).

    Args:
        z:      [N] long — atomic numbers for all atoms.
        batch:  [N] long mapping atom -> frame (or None for a single graph).
        shifts: dict from atomic number to per-atom offset, as produced by
                `fit_atomic_reference`.

    Returns:
        If `batch is None`: scalar Tensor — sum over all atoms.
        Else:               [B] Tensor — per-frame sums via scatter-add.
    """
    # Lookup table indexed by Z. Sized large enough for any element seen.
    z_max = int(z.max()) + 1
    table = torch.zeros(z_max, dtype=torch.get_default_dtype(), device=z.device)
    for atomic_number, shift in shifts.items():
        if atomic_number < z_max:
            table[atomic_number] = shift
    per_atom_shift = table[z]  # [N]

    if batch is None:
        return per_atom_shift.sum()
    n_frames = int(batch.max()) + 1
    out = torch.zeros(n_frames, dtype=per_atom_shift.dtype, device=per_atom_shift.device)
    return out.index_add_(0, batch, per_atom_shift)


def energy_force_loss(
    pred_e: Tensor,
    true_e: Tensor,
    pred_f: Tensor,
    true_f: Tensor,
    n_atoms: Tensor,
    *,
    w_e: float = 1.0,
    w_f: float = 100.0,
    w_s: float = 0.0,
    pred_stress: Tensor | None = None,
    true_stress: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Weighted per-atom energy MSE + force MSE + (optional) stress MSE.

    L = w_e * MSE(E_pred/N, E_true/N) + w_f * MSE(F_pred, F_true) + w_s * MSE(σ_pred, σ_true)

    Per-atom normalization on energy keeps the loss scale-free across system
    sizes. Stress is already intensive (eV/Å³ when ASE units are used) so no
    normalization is applied. When w_s=0 the stress branch is a no-op and the
    function is bit-identical to the pre-PBC API; nb04 and nb05 callers are
    unaffected.

    Args:
        pred_e, true_e:     [B] energies (already shifted by atomic ref).
        pred_f, true_f:     [N_total, 3] forces (concatenated across batch).
        n_atoms:            [B] long, per-frame atom count.
        w_e, w_f, w_s:      scalar weights on each loss term.
        pred_stress, true_stress: optional [B, 3, 3] stress tensors. Required
                                  when w_s>0; ignored when w_s=0.

    Returns:
        (loss, metrics). metrics has keys: loss, energy_mae, force_mae.
        When w_s>0 (i.e. stresses were provided), stress_mae is added.
    """
    per_atom_pred = pred_e / n_atoms.to(pred_e.dtype)
    per_atom_true = true_e / n_atoms.to(true_e.dtype)

    energy_mse = ((per_atom_pred - per_atom_true) ** 2).mean()
    force_mse = ((pred_f - true_f) ** 2).mean()

    loss = w_e * energy_mse + w_f * force_mse
    metrics = {
        "loss": 0.0,  # filled in after stress branch
        "energy_mae": float((per_atom_pred - per_atom_true).abs().mean().detach()),
        "force_mae": float((pred_f - true_f).abs().mean().detach()),
    }

    if w_s > 0:
        if pred_stress is None or true_stress is None:
            raise ValueError("w_s>0 requires pred_stress and true_stress")
        stress_mse = ((pred_stress - true_stress) ** 2).mean()
        loss = loss + w_s * stress_mse
        metrics["stress_mae"] = float((pred_stress - true_stress).abs().mean().detach())

    metrics["loss"] = float(loss.detach())
    return loss, metrics


def _step(
    model,
    batch: dict[str, Tensor],
    *,
    shifts: dict[int, float],
    w_e: float,
    w_f: float,
    w_s: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """Single forward pass: energy + autograd forces (+ stress when requested) + weighted loss.

    Used by both train_one_epoch (with backward) and evaluate (without).

    When w_s > 0 and the batch contains a "stress" key, routes through
    compute_forces_and_stress so that the virial / cell gradient is computed
    via autograd over the strain variable. Otherwise falls back to the original
    pos.requires_grad_ path used by nb04/nb05 — bit-identical behaviour when
    w_s == 0.

    We call .sum() on the per-frame energies in the non-stress path before
    passing to compute_forces because compute_forces wants a scalar; per-frame
    forces are recovered correctly because the disjoint-union batching
    guarantees no cross-frame edges, so each atom's force only depends on its
    own frame's energy.
    """
    graph = batch["graph"]

    if w_s > 0 and "stress" in batch:
        # PBC + stress path: strain perturbation handled inside compute_forces_and_stress
        pred_e_residual, pred_f, pred_s = compute_forces_and_stress(model, graph, create_graph=True)
    else:
        # Original non-stress path: pos must be a leaf
        graph.pos.requires_grad_(True)
        pred_e_residual = model(graph)  # [B] (model only learns residuals)
        pred_f = compute_forces(pred_e_residual.sum(), graph.pos)  # [N_total, 3]
        pred_s = None

    # Subtract the atomic reference from the true energy so model targets are
    # the small residuals matching the model's output.
    ref = apply_atomic_reference(graph.z, graph.batch, shifts).to(batch["energy"].dtype)
    true_residual = batch["energy"] - ref

    loss, metrics = energy_force_loss(
        pred_e_residual,
        true_residual,
        pred_f,
        batch["forces"],
        batch["n_atoms"],
        w_e=w_e,
        w_f=w_f,
        w_s=w_s,
        pred_stress=pred_s,
        true_stress=batch.get("stress") if w_s > 0 else None,
    )
    return loss, metrics


def train_one_epoch(
    model,
    loader,
    optimizer,
    *,
    shifts: dict[int, float],
    w_e: float = 1.0,
    w_f: float = 100.0,
    w_s: float = 0.0,
) -> dict[str, float]:
    """One pass over `loader` with parameter updates. Returns mean metrics."""
    model.train()
    sums = {"loss": 0.0, "energy_mae": 0.0, "force_mae": 0.0}
    if w_s > 0:
        sums["stress_mae"] = 0.0
    n_batches = 0
    for batch in loader:
        optimizer.zero_grad()
        loss, metrics = _step(model, batch, shifts=shifts, w_e=w_e, w_f=w_f, w_s=w_s)
        loss.backward()
        optimizer.step()
        for k in sums:
            sums[k] += metrics[k]
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in sums.items()}


def evaluate(
    model,
    loader,
    *,
    shifts: dict[int, float],
    w_e: float = 1.0,
    w_f: float = 100.0,
    w_s: float = 0.0,
) -> dict[str, float]:
    """One pass over `loader` without parameter updates. Returns mean metrics.

    We do NOT use torch.no_grad() here: forces require autograd through pos.
    Model parameter grads still flow during this pass, but we never call
    optimizer.step(), so nothing actually gets updated. The cost is small for
    rMD17-sized molecules.
    """
    model.eval()
    sums = {"loss": 0.0, "energy_mae": 0.0, "force_mae": 0.0}
    if w_s > 0:
        sums["stress_mae"] = 0.0
    n_batches = 0
    for batch in loader:
        loss, metrics = _step(model, batch, shifts=shifts, w_e=w_e, w_f=w_f, w_s=w_s)
        for k in sums:
            sums[k] += metrics[k]
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in sums.items()}


def train(
    model,
    train_loader,
    val_loader,
    *,
    n_epochs: int,
    lr: float,
    w_e: float = 1.0,
    w_f: float = 100.0,
    w_s: float = 0.0,
    shifts: dict[int, float],
) -> pl.DataFrame:
    """Train `model` for `n_epochs` and return a polars run-log DataFrame.

    Columns: epoch (int), split (str: "train"/"val"), loss, energy_mae, force_mae.
    Two rows per epoch (one per split). When w_s>0 a stress_mae column is also
    present. Caller plots straight from this frame.

    Optimizer is Adam(lr). LR scheduling, EMA, gradient clipping are deliberately
    omitted for clarity — production tricks belong in a follow-up.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows: list[dict[str, float | int | str]] = []
    for epoch in range(n_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, shifts=shifts, w_e=w_e, w_f=w_f, w_s=w_s
        )
        val_metrics = evaluate(model, val_loader, shifts=shifts, w_e=w_e, w_f=w_f, w_s=w_s)
        rows.append({"epoch": epoch, "split": "train", **train_metrics})
        rows.append({"epoch": epoch, "split": "val", **val_metrics})
    return pl.DataFrame(rows)
