"""Tests for tinymlip.forces."""

from __future__ import annotations

from dataclasses import replace

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


def test_compute_forces_and_stress_rejects_non_pbc_graph():
    """A non-PBC graph has cell=None; the function must raise a clear error
    rather than crashing on a NoneType attribute access."""
    import pytest
    from ase import Atoms

    from tinymlip.forces import compute_forces_and_stress
    from tinymlip.models import EquivariantMPNN

    # Molecule (no cell, no PBC)
    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    g = build_graph(atoms, cutoff=2.0)
    assert g.cell is None  # sanity

    torch.manual_seed(0)
    model = EquivariantMPNN(n_layers=1, hidden_dim=8, num_basis=8, cutoff=2.0)

    with pytest.raises(ValueError, match="requires a PBC graph"):
        compute_forces_and_stress(model, g)


def test_compute_forces_and_stress_autograd_matches_numerical_strain_derivative():
    """The strain-derivative formula σ = (1/V) ∂E/∂ε must match a central
    finite-difference of the model energy under a manual strain on cell+pos."""
    from ase import Atoms

    from tinymlip.forces import compute_forces_and_stress

    # Small FCC-Cu-like 4-atom cubic cell, random perturbation
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
    torch.manual_seed(0)
    g = build_graph(atoms, cutoff=4.0, dtype=torch.float64)

    model = EquivariantMPNN(n_layers=2, hidden_dim=8, num_basis=8, cutoff=4.0).double()

    # Autograd stress
    E_ad, F_ad, sigma_ad = compute_forces_and_stress(model, g)  # noqa: N806 — physics notation

    # Numerical: deform manually for each independent strain component, ε=±h.
    h = 1e-4

    def E_at_strain(eps: torch.Tensor) -> torch.Tensor:  # noqa: N802 — E is standard for energy
        # eps: [3, 3] symmetric
        pos_def = g.pos + g.pos @ eps
        cell_def = g.cell + g.cell @ eps
        g_def = replace(g, pos=pos_def, cell=cell_def)
        return model(g_def).sum()

    sigma_num = torch.zeros(3, 3, dtype=torch.float64)
    V = g.cell.det().abs().item()  # noqa: N806 — V for volume
    for i in range(3):
        for j in range(3):
            eps_p = torch.zeros(3, 3, dtype=torch.float64)
            eps_m = torch.zeros(3, 3, dtype=torch.float64)
            # symmetric strain perturbation
            eps_p[i, j] += h
            eps_p[j, i] += h
            eps_m[i, j] -= h
            eps_m[j, i] -= h
            E_p = E_at_strain(eps_p)  # noqa: N806 — E for energy
            E_m = E_at_strain(eps_m)  # noqa: N806 — E for energy
            # ∂E/∂ε_ij where ε is symmetric and we perturbed both (i,j) and (j,i)
            # — so the finite-difference numerator already counts both, and we
            # divide by 2h to recover the single-component derivative.
            sigma_num[i, j] = (E_p - E_m) / (4 * h) / V
    sigma_num = 0.5 * (sigma_num + sigma_num.T)

    torch.testing.assert_close(sigma_ad, sigma_num, atol=1e-5, rtol=1e-3)


def test_compute_forces_and_stress_sign_convention_matches_ase_emt():
    """Run a hand-built pair-LJ "model" on a Cu config and confirm the autograd
    stress is finite, symmetric in cubic symmetry, and has uniform-sign diagonal.

    We construct a closed-form pair-LJ energy over edges so we don't rely on a
    trained network. The point isn't to match EMT numerically (LJ ≠ EMT), it's
    to confirm that on a SAME-pair-potential system, our autograd σ has the
    right shape, finite values, and a sign that respects the cubic symmetry of
    the FCC unit cell."""
    from ase import Atoms

    from tinymlip.forces import compute_forces_and_stress

    a = 3.6
    atoms = Atoms(
        numbers=[1] * 4,
        positions=[[0, 0, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True,
    )
    g = build_graph(atoms, cutoff=4.0, dtype=torch.float64)

    # Pure pair-LJ "model": takes graph -> scalar E.
    class LJ:
        def __call__(self, graph):
            src, dst = graph.edge_index
            edge_vec = graph.pos[dst] - graph.pos[src]
            if graph.shift_idx is not None:
                shift_f = graph.shift_idx.to(edge_vec.dtype)
                edge_vec = edge_vec + shift_f @ graph.cell
            r = edge_vec.norm(dim=-1).clamp(min=1e-6)
            sigma_lj = 2.5
            eps_lj = 0.1
            sr6 = (sigma_lj / r) ** 6
            E = 0.5 * (4 * eps_lj * (sr6**2 - sr6)).sum()  # noqa: N806 — physics notation
            return E

    E, F, sigma_ad = compute_forces_and_stress(LJ(), g)  # noqa: N806 — physics notation

    diag = torch.diagonal(sigma_ad)
    assert diag.shape == (3,)
    assert torch.isfinite(F).all()
    assert torch.isfinite(sigma_ad).all()
    # All three diagonal entries should have the same sign for an isotropic
    # cubic LJ system — that's the rotation/cubic symmetry of the problem.
    assert (diag.sign().abs() == 1).all()
    assert (diag.sign() == diag.sign()[0]).all()
