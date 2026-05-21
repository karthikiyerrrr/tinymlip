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
