"""Tests for tinymlip.forces."""

from __future__ import annotations

import torch

from tinymlip.forces import compute_forces


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
