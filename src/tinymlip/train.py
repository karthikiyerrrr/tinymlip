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
import torch  # noqa: F401
from torch import Tensor  # noqa: F401

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
        raise ValueError(
            f"structures and energies disagree: {len(structures)} vs {len(energies)}"
        )
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
    w, *_ = np.linalg.lstsq(
        design_matrix, np.asarray(energies, dtype=np.float64), rcond=None
    )
    return {int(z): float(w[i]) for i, z in enumerate(elements)}
