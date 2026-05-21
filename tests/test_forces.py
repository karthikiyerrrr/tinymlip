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


def test_invariant_model_forces_finite(ethanol_atoms):
    """Forces from InvariantMPNN are finite (no NaN/inf from clamp or basis)."""
    torch.manual_seed(0)
    model = InvariantMPNN(hidden_dim=16, num_basis=8, cutoff=5.0, n_layers=2)

    graph = build_graph(ethanol_atoms, cutoff=5.0)
    graph.pos.requires_grad_(True)
    energy = model(graph)
    forces = compute_forces(energy, graph.pos)

    assert forces.shape == graph.pos.shape
    assert torch.isfinite(forces).all()


def test_equivariant_model_forces_finite(ethanol_atoms):
    """Forces from EquivariantMPNN are finite.

    Equivariant failure mode is different from invariant (Vv.norm with v=0 at
    the start can produce NaN gradients if epsilon-less norm is used wrong).
    Kept as a separate test so a failure points at the right side.
    """
    torch.manual_seed(0)
    model = EquivariantMPNN(hidden_dim=16, num_basis=8, cutoff=5.0, n_layers=2)

    graph = build_graph(ethanol_atoms, cutoff=5.0)
    graph.pos.requires_grad_(True)
    energy = model(graph)
    forces = compute_forces(energy, graph.pos)

    assert forces.shape == graph.pos.shape
    assert torch.isfinite(forces).all()


def test_invariant_forces_match_numerical_gradient(ethanol_atoms):
    """Smoking gun for F = -grad(E): autograd forces equal central differences.

    Pick atom 0 axis 0; perturb by eps = 1e-3 A and recompute energy at each
    side. Tolerance 1e-3 absolute: float32 central-difference error is
    ~eps^2 (truncation) + eps_machine/eps ~ 1e-4 in float32, plus model
    nonlinearity scale. Use float64 to keep this clean.

    Runtime: ~1-3s on CPU (three forward passes through a 2-layer model on
    9 atoms). Allowed to be slower than the suite average -- see spec's
    'Tests' section.
    """
    torch.manual_seed(0)
    cutoff = 5.0
    model = InvariantMPNN(
        hidden_dim=16, num_basis=8, cutoff=cutoff, n_layers=2
    ).double()  # float64 for clean finite differences

    # Autograd path.
    graph = build_graph(ethanol_atoms, cutoff=cutoff, dtype=torch.float64)
    graph.pos.requires_grad_(True)
    energy = model(graph)
    forces = compute_forces(energy, graph.pos)
    autograd_force = forces[0, 0].item()

    # Numerical path: rebuild the graph from perturbed ASE positions so
    # the connectivity is recomputed (won't change at eps=1e-3, but safer).
    eps = 1e-3

    atoms_plus = ethanol_atoms.copy()
    pos_plus = atoms_plus.get_positions()
    pos_plus[0, 0] += eps
    atoms_plus.set_positions(pos_plus)
    graph_plus = build_graph(atoms_plus, cutoff=cutoff, dtype=torch.float64)
    e_plus = model(graph_plus).item()

    atoms_minus = ethanol_atoms.copy()
    pos_minus = atoms_minus.get_positions()
    pos_minus[0, 0] -= eps
    atoms_minus.set_positions(pos_minus)
    graph_minus = build_graph(atoms_minus, cutoff=cutoff, dtype=torch.float64)
    e_minus = model(graph_minus).item()

    numerical_force = -(e_plus - e_minus) / (2 * eps)

    assert abs(numerical_force - autograd_force) < 1e-3, (
        f"numerical={numerical_force:.6f}, autograd={autograd_force:.6f}"
    )


def test_equivariant_forces_match_numerical_gradient(ethanol_atoms):
    """Smoking gun for F = -grad(E) on the equivariant model.

    Same protocol as the invariant test above: pick atom 0 axis 0, perturb
    by eps = 1e-3 A, compare autograd force to central-difference numerical
    force. Float64 throughout to keep the comparison clean. Tolerance and
    runtime characteristics match the invariant version.
    """
    torch.manual_seed(0)
    cutoff = 5.0
    model = EquivariantMPNN(hidden_dim=16, num_basis=8, cutoff=cutoff, n_layers=2).double()

    graph = build_graph(ethanol_atoms, cutoff=cutoff, dtype=torch.float64)
    graph.pos.requires_grad_(True)
    energy = model(graph)
    forces = compute_forces(energy, graph.pos)
    autograd_force = forces[0, 0].item()

    eps = 1e-3

    atoms_plus = ethanol_atoms.copy()
    pos_plus = atoms_plus.get_positions()
    pos_plus[0, 0] += eps
    atoms_plus.set_positions(pos_plus)
    graph_plus = build_graph(atoms_plus, cutoff=cutoff, dtype=torch.float64)
    e_plus = model(graph_plus).item()

    atoms_minus = ethanol_atoms.copy()
    pos_minus = atoms_minus.get_positions()
    pos_minus[0, 0] -= eps
    atoms_minus.set_positions(pos_minus)
    graph_minus = build_graph(atoms_minus, cutoff=cutoff, dtype=torch.float64)
    e_minus = model(graph_minus).item()

    numerical_force = -(e_plus - e_minus) / (2 * eps)

    assert abs(numerical_force - autograd_force) < 1e-3, (
        f"numerical={numerical_force:.6f}, autograd={autograd_force:.6f}"
    )
