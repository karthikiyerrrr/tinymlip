"""End-to-end invariance / equivariance properties under PBC + stress."""

from __future__ import annotations

import torch
from ase import Atoms

from tinymlip.forces import compute_forces_and_stress
from tinymlip.graph import build_graph
from tinymlip.models import EquivariantMPNN


def _small_cu_atoms(a: float = 3.6, perturb: float = 0.0) -> Atoms:
    """FCC Cu unit cell (4 atoms).

    `perturb` adds a small random displacement to each atom so forces are
    non-zero — needed for rotation-equivariance tests where the assertion
    ``F_b ≈ R @ F_a`` is trivially satisfied (and numerically meaningless)
    when F_a ≈ 0.
    """
    import numpy as np

    positions = [
        [0.0, 0.0, 0.0],
        [0.0, a / 2, a / 2],
        [a / 2, 0.0, a / 2],
        [a / 2, a / 2, 0.0],
    ]
    if perturb > 0.0:
        rng = np.random.default_rng(42)
        positions = (np.array(positions) + rng.uniform(-perturb, perturb, (4, 3))).tolist()
    return Atoms(
        numbers=[29] * 4,
        positions=positions,
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True,
    )


def test_supercell_extensivity_doubles_energy_preserves_stress():
    """A 2x1x1 supercell has 2x the energy and identical stress."""
    atoms = _small_cu_atoms()
    big = atoms.repeat((2, 1, 1))

    g_small = build_graph(atoms, cutoff=4.0, dtype=torch.float64)
    g_big = build_graph(big, cutoff=4.0, dtype=torch.float64)

    torch.manual_seed(0)
    model = EquivariantMPNN(n_layers=1, hidden_dim=8, num_basis=8, cutoff=4.0).double()

    E_s, _, sigma_s = compute_forces_and_stress(model, g_small)  # noqa: N806 — E physics notation
    E_b, _, sigma_b = compute_forces_and_stress(model, g_big)  # noqa: N806

    torch.testing.assert_close(E_b, 2 * E_s, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sigma_b, sigma_s, atol=1e-5, rtol=1e-4)


def test_stress_rotates_as_R_sigma_RT():  # noqa: N802 — R, sigma, T are standard physics symbols
    """Apply a random rotation to cell + positions; sigma should transform as
    R sigma R^T and forces should rotate as R F.

    The atoms are slightly perturbed from the perfect FCC lattice so that
    forces are non-zero.  A perfectly symmetric FCC cell has F ≈ 0 by
    symmetry; rotating zero forces trivially satisfies F_b ≈ R @ F_a but the
    assertion ``F_b ≈ R @ 0 ≈ 0`` carries no information and can fail due to
    floating-point noise at the 1e-3 level.  A small random displacement of
    ~0.1 Å keeps forces well above 1e-5 eV/Å, making the test meaningful.
    """
    atoms = _small_cu_atoms(perturb=0.1)
    g = build_graph(atoms, cutoff=4.0, dtype=torch.float64)
    torch.manual_seed(1)
    model = EquivariantMPNN(n_layers=1, hidden_dim=8, num_basis=8, cutoff=4.0).double()

    E_a, F_a, sigma_a = compute_forces_and_stress(model, g)  # noqa: N806 — E, F physics notation

    # Random rotation
    A = torch.randn(3, 3, dtype=torch.float64)  # noqa: N806 — A is a matrix (capitalized by convention)
    Q, _ = torch.linalg.qr(A)  # noqa: N806 — Q for orthogonal matrix (QR decomposition)
    # Ensure proper rotation (det +1)
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    R = Q  # noqa: N806 — R for rotation matrix

    rot_atoms = atoms.copy()
    rot_atoms.set_cell(atoms.cell.array @ R.numpy().T, scale_atoms=False)
    rot_atoms.set_positions(atoms.positions @ R.numpy().T)
    g_rot = build_graph(rot_atoms, cutoff=4.0, dtype=torch.float64)

    E_b, F_b, sigma_b = compute_forces_and_stress(model, g_rot)  # noqa: N806

    torch.testing.assert_close(E_b, E_a, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(F_b, F_a @ R.T, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sigma_b, R @ sigma_a @ R.T, atol=1e-5, rtol=1e-5)
