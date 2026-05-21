"""Tests for tinymlip.train: reference-shift fit and energy+force loss."""

from __future__ import annotations

import numpy as np
import pytest
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
        e_pred,
        e_true,
        f_pred,
        f_true,
        n_atoms,
        w_e=1.0,
        w_f=100.0,
    )
    # Per-atom energy MSE = (0.1)**2 = 0.01.  Force MSE = 4.0.
    # Loss = 1.0 * 0.01 + 100.0 * 4.0 = 400.01.
    assert abs(float(loss) - 400.01) < 1e-5
    # MAE: per-atom energy MAE = 0.1; force MAE = 2.0.
    assert abs(metrics["energy_mae"] - 0.1) < 1e-6
    assert abs(metrics["force_mae"] - 2.0) < 1e-6


def test_train_runs_two_epochs_returns_polars_history(ethanol_atoms):
    """Smoke test the full loop: 2 epochs on a tiny synthetic batch.

    What this guards:
      - train returns a polars DataFrame with the expected columns.
      - both split rows (train, val) appear in the history.
      - loss is finite after the run.

    What this deliberately does NOT check:
      - whether the loss went down (brittle on 2 epochs / random init).
    """
    import polars as pl
    from torch.utils.data import DataLoader

    from tinymlip.data import make_collate
    from tinymlip.models import InvariantMPNN
    from tinymlip.train import fit_atomic_reference, train

    torch.manual_seed(0)

    # Build a tiny dataset: 4 ethanol frames with synthetic energies/forces.
    # We do not need the labels to be physically right — we just need the loop
    # to execute end-to-end and update parameters.
    samples = []
    for _ in range(4):
        atoms = ethanol_atoms.copy()
        atoms.set_positions(atoms.get_positions() + 0.01 * torch.randn(9, 3).numpy())
        samples.append(
            {
                "z": torch.as_tensor(atoms.numbers, dtype=torch.long),
                "pos": torch.as_tensor(atoms.positions, dtype=torch.float32),
                "energy": torch.tensor(-4209.0, dtype=torch.float32),
                "forces": torch.zeros(9, 3, dtype=torch.float32),
                "frame_idx": torch.tensor(0, dtype=torch.long),
            }
        )

    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    ds = _ListDataset(samples)
    collate = make_collate(cutoff=5.0)
    train_loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)

    structures = [Atoms(numbers=s["z"].numpy(), positions=s["pos"].numpy()) for s in samples]
    energies = torch.stack([s["energy"] for s in samples]).numpy()
    shifts = fit_atomic_reference(structures, energies)

    model = InvariantMPNN(hidden_dim=16, num_basis=8, cutoff=5.0, n_layers=2)

    history = train(
        model,
        train_loader,
        val_loader,
        n_epochs=2,
        lr=1e-3,
        w_e=1.0,
        w_f=100.0,
        shifts=shifts,
    )

    assert isinstance(history, pl.DataFrame)
    assert set(history.columns) == {"epoch", "split", "loss", "energy_mae", "force_mae"}
    # 2 epochs * 2 splits = 4 rows.
    assert history.height == 4
    splits = set(history["split"].to_list())
    assert splits == {"train", "val"}
    # All losses finite.
    assert all(np.isfinite(v) for v in history["loss"].to_list())


def test_energy_force_loss_with_w_s_zero_matches_old_behavior():
    """When w_s=0 the new stress-aware loss must equal the original output."""
    from tinymlip.train import energy_force_loss

    pred_e = torch.tensor([1.2, 0.8])
    true_e = torch.tensor([1.0, 1.0])
    pred_f = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    true_f = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    n_atoms = torch.tensor([1, 1])

    loss_baseline, m_baseline = energy_force_loss(
        pred_e, true_e, pred_f, true_f, n_atoms, w_e=1.0, w_f=100.0
    )
    pred_s = torch.zeros(2, 3, 3)
    true_s = torch.zeros(2, 3, 3)
    loss_new, m_new = energy_force_loss(
        pred_e,
        true_e,
        pred_f,
        true_f,
        n_atoms,
        w_e=1.0,
        w_f=100.0,
        w_s=0.0,
        pred_stress=pred_s,
        true_stress=true_s,
    )
    torch.testing.assert_close(loss_baseline, loss_new)
    assert m_new["energy_mae"] == m_baseline["energy_mae"]


def test_energy_force_loss_with_w_s_nonzero_includes_stress_term():
    """With w_s>0, a nonzero stress error increases the loss."""
    from tinymlip.train import energy_force_loss

    pred_e = torch.tensor([0.0])
    true_e = torch.tensor([0.0])
    pred_f = torch.zeros(2, 3)
    true_f = torch.zeros(2, 3)
    n_atoms = torch.tensor([2])

    pred_s = torch.zeros(1, 3, 3)
    true_s = torch.ones(1, 3, 3)  # MSE = 1

    loss, m = energy_force_loss(
        pred_e,
        true_e,
        pred_f,
        true_f,
        n_atoms,
        w_e=1.0,
        w_f=100.0,
        w_s=2.0,
        pred_stress=pred_s,
        true_stress=true_s,
    )
    assert float(loss) == pytest.approx(2.0, abs=1e-6)
    assert "stress_mae" in m
    assert m["stress_mae"] == pytest.approx(1.0, abs=1e-6)
