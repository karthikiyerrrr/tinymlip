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


def test_energy_force_loss_zero_at_truth():
    """pred == true => loss is exactly 0 and both MAEs are 0."""
    from tinymlip.train import energy_force_loss

    e = torch.tensor([1.0, -2.0, 3.5])
    f = torch.randn(3 * 9, 3)
    n_atoms = torch.tensor([9, 9, 9], dtype=torch.long)
    loss, metrics = energy_force_loss(e, e.clone(), f, f.clone(), n_atoms)
    assert float(loss) == 0.0
    assert metrics["energy_mae"] == 0.0
    assert metrics["force_mae"] == 0.0


def test_energy_force_loss_components_combine_with_weights():
    """Build a case with known E error and known F error; check the weighted sum."""
    from tinymlip.train import energy_force_loss

    # Energies: pred = true + 1 per frame, n_atoms=10 per frame, so per-atom
    # error is 0.1 and per-atom MSE is 0.01. Three frames => same.
    n_atoms = torch.tensor([10, 10, 10], dtype=torch.long)
    e_true = torch.tensor([0.0, 0.0, 0.0])
    e_pred = e_true + 1.0  # per-atom: 0.1
    # Forces: pred - true = 2 along every component. MSE = 4.
    f_true = torch.zeros(30, 3)
    f_pred = f_true + 2.0
    loss, metrics = energy_force_loss(
        e_pred, e_true, f_pred, f_true, n_atoms, w_e=1.0, w_f=100.0,
    )
    # Per-atom energy MSE = (0.1)**2 = 0.01.  Force MSE = 4.0.
    # Loss = 1.0 * 0.01 + 100.0 * 4.0 = 400.01.
    assert abs(float(loss) - 400.01) < 1e-5
    # MAE: per-atom energy MAE = 0.1; force MAE = 2.0.
    assert abs(metrics["energy_mae"] - 0.1) < 1e-6
    assert abs(metrics["force_mae"] - 2.0) < 1e-6
