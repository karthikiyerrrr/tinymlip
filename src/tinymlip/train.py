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
import polars as pl  # noqa: F401
import torch
from torch import Tensor

from tinymlip.forces import compute_forces  # noqa: F401


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
) -> tuple[Tensor, dict[str, float]]:
    """Weighted per-atom energy MSE + force MSE, with MAEs returned for logging.

    L = w_e * MSE(E_pred / N, E_true / N) + w_f * MSE(F_pred, F_true)

    Per-atom normalization on energy makes the loss scale-free across system
    sizes — important once nb 06 mixes 9-atom ethanols with larger crystals.
    The default w_f=100 balances per-component magnitudes: post-shift energies
    are ~1 kcal/mol while force components are ~50 kcal/mol/Å.

    Args:
        pred_e: [B] predicted energies (already shifted — i.e. residuals).
        true_e: [B] target energies (already shifted by `apply_atomic_reference`).
        pred_f: [N_total, 3] predicted forces (concatenated across the batch).
        true_f: [N_total, 3] target forces (same layout).
        n_atoms: [B] long, per-frame atom count for per-atom normalization.
        w_e, w_f: scalar weights on the energy and force terms.

    Returns:
        (loss, metrics_dict) where metrics_dict has the keys:
            loss         — the scalar loss as a float (a `.item()` copy).
            energy_mae   — mean absolute error on PER-ATOM energy (same units as
                           pred_e / n_atoms — kcal/mol/atom for rMD17).
            force_mae    — mean absolute error on force components.
    """
    per_atom_pred = pred_e / n_atoms.to(pred_e.dtype)
    per_atom_true = true_e / n_atoms.to(true_e.dtype)

    energy_mse = ((per_atom_pred - per_atom_true) ** 2).mean()
    force_mse = ((pred_f - true_f) ** 2).mean()

    loss = w_e * energy_mse + w_f * force_mse

    metrics = {
        "loss": float(loss.detach()),
        "energy_mae": float((per_atom_pred - per_atom_true).abs().mean().detach()),
        "force_mae": float((pred_f - true_f).abs().mean().detach()),
    }
    return loss, metrics
