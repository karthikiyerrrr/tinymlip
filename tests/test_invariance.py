"""Model-level rotation-invariance tests for the full MPNNs.

Layer-level rotation tests live in test_layers.py (InvariantInteraction is
invariant; EquivariantInteraction's v rotates with the molecule). These
model-level tests check that the composition end-to-end — embedding,
stacked interactions, atomic readout, and per-atom sum — preserves the
right symmetry.
"""

from __future__ import annotations

import numpy as np
import torch

from tinymlip.graph import build_graph
from tinymlip.models import EquivariantMPNN, InvariantMPNN


def _random_rotation(seed: int = 0) -> torch.Tensor:
    """Random proper rotation matrix in float64. det(R) = +1."""
    rng = np.random.default_rng(seed)
    A = torch.from_numpy(rng.standard_normal((3, 3)))  # noqa: N806 — matrix
    q, _ = torch.linalg.qr(A)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(torch.float64)


def test_invariant_mpnn_rotation_invariance(ethanol_atoms):
    """InvariantMPNN energy is unchanged under a rigid rotation of positions."""
    torch.manual_seed(0)
    cutoff = 5.0
    model = InvariantMPNN(hidden_dim=16, num_basis=8, cutoff=cutoff, n_layers=2).double()

    graph = build_graph(ethanol_atoms, cutoff=cutoff, dtype=torch.float64)
    e_orig = model(graph)

    R = _random_rotation(seed=1)  # noqa: N806 — R is standard rotation notation
    rotated = ethanol_atoms.copy()
    rotated.set_positions(rotated.get_positions() @ R.numpy().T)
    graph_rot = build_graph(rotated, cutoff=cutoff, dtype=torch.float64)
    e_rot = model(graph_rot)

    assert torch.allclose(e_orig, e_rot, atol=1e-10), (
        f"InvariantMPNN energy drifted under rotation: {e_orig.item()} vs {e_rot.item()}"
    )


def test_equivariant_mpnn_rotation_invariance(ethanol_atoms):
    """EquivariantMPNN energy is unchanged under a rigid rotation of positions.

    Vector features rotate with the molecule, but the readout consumes only
    the scalar channel s, so the final energy is rotation-invariant.
    """
    torch.manual_seed(0)
    cutoff = 5.0
    model = EquivariantMPNN(hidden_dim=16, num_basis=8, cutoff=cutoff, n_layers=2).double()

    graph = build_graph(ethanol_atoms, cutoff=cutoff, dtype=torch.float64)
    e_orig = model(graph)

    R = _random_rotation(seed=2)  # noqa: N806
    rotated = ethanol_atoms.copy()
    rotated.set_positions(rotated.get_positions() @ R.numpy().T)
    graph_rot = build_graph(rotated, cutoff=cutoff, dtype=torch.float64)
    e_rot = model(graph_rot)

    assert torch.allclose(e_orig, e_rot, atol=1e-8), (
        f"EquivariantMPNN energy drifted under rotation: {e_orig.item()} vs {e_rot.item()}"
    )


def test_pbc_translation_invariance_predicted_energy_unchanged():
    """Translating ALL atoms by an arbitrary vector (not necessarily a lattice
    vector) under PBC must leave the predicted energy unchanged."""
    from ase import Atoms

    a = 3.6
    atoms = Atoms(
        numbers=[29] * 4,
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, a / 2, a / 2],
            [a / 2, 0.0, a / 2],
            [a / 2, a / 2, 0.0],
        ],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True,
    )
    shifted = atoms.copy()
    shifted.translate([0.37, -1.21, 0.04])  # arbitrary vector

    g1 = build_graph(atoms, cutoff=4.0, dtype=torch.float64)
    g2 = build_graph(shifted, cutoff=4.0, dtype=torch.float64)

    torch.manual_seed(0)
    model = EquivariantMPNN(n_layers=1, hidden_dim=8, num_basis=8, cutoff=4.0).double()
    with torch.no_grad():
        e1 = model(g1)
        e2 = model(g2)
    torch.testing.assert_close(e1, e2, atol=1e-6, rtol=1e-6)


def test_pbc_permutation_invariance_predicted_energy_unchanged():
    """Reordering atoms (and their per-atom data consistently) under PBC must
    leave the predicted energy unchanged."""
    from ase import Atoms

    a = 3.6
    atoms = Atoms(
        numbers=[29, 29, 29, 29],
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, a / 2, a / 2],
            [a / 2, 0.0, a / 2],
            [a / 2, a / 2, 0.0],
        ],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True,
    )
    perm = np.array([2, 0, 3, 1])
    permuted = Atoms(
        numbers=np.asarray(atoms.numbers)[perm],
        positions=atoms.positions[perm],
        cell=atoms.cell.array,
        pbc=True,
    )
    g1 = build_graph(atoms, cutoff=4.0, dtype=torch.float64)
    g2 = build_graph(permuted, cutoff=4.0, dtype=torch.float64)

    torch.manual_seed(0)
    model = EquivariantMPNN(n_layers=1, hidden_dim=8, num_basis=8, cutoff=4.0).double()
    with torch.no_grad():
        e1 = model(g1)
        e2 = model(g2)
    torch.testing.assert_close(e1, e2, atol=1e-6, rtol=1e-6)
