"""Tests for tinymlip.train: reference-shift fit and energy+force loss."""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms

from tinymlip.train import fit_atomic_reference


def test_reference_shift_recovers_known_offsets():
    """If energies are exactly a linear combination of atom counts, the fit
    must recover the per-element offsets to machine precision.

    Tiny synthetic dataset: 3 fake molecules with known H/C/O composition,
    energies = sum_i shift_z[z_i] for hand-picked shifts. The fit must return
    those same shifts.
    """
    rng = np.random.default_rng(0)
    true_shift = {1: -0.5, 6: -38.0, 8: -75.0}  # H, C, O

    structures: list[Atoms] = []
    energies = []
    # 30 small molecules with random (H, C, O) compositions in {1..6} atoms each.
    for _ in range(30):
        n_h, n_c, n_o = rng.integers(1, 6, size=3).tolist()
        numbers = [1] * n_h + [6] * n_c + [8] * n_o
        # Positions don't matter for the linear-regression fit — only counts do.
        positions = rng.normal(size=(len(numbers), 3))
        structures.append(Atoms(numbers=numbers, positions=positions))
        energies.append(n_h * true_shift[1] + n_c * true_shift[6] + n_o * true_shift[8])

    fitted = fit_atomic_reference(structures, np.array(energies))

    for z, ref in true_shift.items():
        assert abs(fitted[z] - ref) < 1e-6, f"z={z}: fitted {fitted[z]} vs true {ref}"


def test_apply_atomic_reference_single_frame():
    """Single graph (batch=None): returns scalar = sum of shifts over atoms."""
    from tinymlip.train import apply_atomic_reference

    shifts = {1: -0.5, 6: -38.0, 8: -75.0}
    z = torch.tensor([6, 1, 1, 1, 8], dtype=torch.long)  # CH3O
    expected = -38.0 + 3 * (-0.5) + (-75.0)  # = -114.5

    out = apply_atomic_reference(z, batch=None, shifts=shifts)
    assert out.dim() == 0
    assert abs(float(out) - expected) < 1e-6


def test_apply_atomic_reference_batched():
    """Batched (batch=[N]): returns [B] per-frame shifts."""
    from tinymlip.train import apply_atomic_reference

    shifts = {1: -0.5, 6: -38.0, 8: -75.0}
    # Frame 0: CH3O = -38 - 1.5 - 75 = -114.5
    # Frame 1: H2O  = -75 - 1.0 = -76.0
    z = torch.tensor([6, 1, 1, 1, 8, 8, 1, 1], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)

    out = apply_atomic_reference(z, batch=batch, shifts=shifts)
    assert out.shape == (2,)
    assert abs(float(out[0]) - (-114.5)) < 1e-6
    assert abs(float(out[1]) - (-76.0)) < 1e-6
