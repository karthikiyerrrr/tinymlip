"""Tests for tinymlip.forces."""

from __future__ import annotations

import torch

from tinymlip.forces import compute_forces
from tinymlip.graph import build_graph
from tinymlip.models import EquivariantMPNN, InvariantMPNN


def test_compute_forces_negates_analytic_gradient():
    """For E = sum(pos**2), dE/dpos = 2*pos, so F = -2*pos exactly."""
    pos = torch.tensor([[1.0, 2.0, 3.0], [-0.5, 0.0, 4.0]], requires_grad=True)
    energy = (pos**2).sum()  # scalar
    forces = compute_forces(energy, pos)

    expected = -2.0 * pos.detach()
    assert forces.shape == pos.shape
    assert torch.allclose(forces, expected, atol=1e-6)


def test_compute_forces_create_graph_lets_us_backprop():
    """create_graph=True must be set so a force-matching loss can backprop."""
    pos = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    energy = (pos**3).sum()
    forces = compute_forces(energy, pos)

    # If create_graph=False, this .backward() would raise:
    #   RuntimeError: element 0 of tensors does not require grad ...
    loss = (forces**2).sum()
    loss.backward()
    assert pos.grad is not None


def test_invariant_mpnn_forward_returns_scalar(ethanol_atoms):
    """Forward pass returns a single scalar energy with a grad path to pos."""
    torch.manual_seed(0)
    model = InvariantMPNN(hidden_dim=16, num_basis=8, cutoff=5.0, n_layers=2)

    graph = build_graph(ethanol_atoms, cutoff=5.0)
    graph.pos.requires_grad_(True)
    energy = model(graph)

    assert energy.dim() == 0  # scalar
    assert energy.requires_grad  # gradient path to pos is intact


def test_equivariant_mpnn_forward_returns_scalar(ethanol_atoms):
    """Forward pass returns a single scalar energy with a grad path to pos."""
    torch.manual_seed(0)
    model = EquivariantMPNN(hidden_dim=16, num_basis=8, cutoff=5.0, n_layers=2)

    graph = build_graph(ethanol_atoms, cutoff=5.0)
    graph.pos.requires_grad_(True)
    energy = model(graph)

    assert energy.dim() == 0
    assert energy.requires_grad
