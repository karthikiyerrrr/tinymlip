"""Tests for graph batching: collate_graphs disjoint-union + batched model equivalence."""

from __future__ import annotations

import pytest
import torch

from tinymlip.graph import build_graph, collate_graphs
from tinymlip.models import EquivariantMPNN, InvariantMPNN


def _three_ethanols(ethanol_atoms):
    """Build three independent ethanol graphs (fixture is mutated freshly each call)."""
    # Tiny perturbations so the three graphs are not literally identical tensors.
    # Connectivity stays the same; this just avoids any accidental aliasing in tests.
    graphs = []
    for shift in (0.0, 0.01, -0.01):
        atoms = ethanol_atoms.copy()
        pos = atoms.get_positions() + shift
        atoms.set_positions(pos)
        graphs.append(build_graph(atoms, cutoff=5.0))
    return graphs


def test_collate_graphs_shapes(ethanol_atoms):
    """Concatenated atom counts and edge counts; batch vector is [0..0, 1..1, 2..2]."""
    graphs = _three_ethanols(ethanol_atoms)
    n_atoms_each = graphs[0].n_atoms  # 9 for ethanol
    n_edges_each = graphs[0].n_edges

    batched = collate_graphs(graphs)

    assert batched.n_atoms == 3 * n_atoms_each
    assert batched.n_edges == 3 * n_edges_each
    assert batched.batch is not None
    expected_batch = torch.cat(
        [
            torch.zeros(n_atoms_each, dtype=torch.long),
            torch.ones(n_atoms_each, dtype=torch.long),
            torch.full((n_atoms_each,), 2, dtype=torch.long),
        ]
    )
    assert torch.equal(batched.batch, expected_batch)
    # Edge indices must point inside the [0, 3*N) range.
    assert int(batched.edge_index.max()) < batched.n_atoms


def test_collate_no_cross_edges(ethanol_atoms):
    """No edge connects atoms in different frames. The load-bearing correctness check."""
    graphs = _three_ethanols(ethanol_atoms)
    batched = collate_graphs(graphs)
    src, dst = batched.edge_index
    assert torch.equal(batched.batch[src], batched.batch[dst]), (
        "collate_graphs leaked an edge between two different frames"
    )


def test_collate_graphs_pbc_stacks_cells_and_concats_shift_idx():
    """Two PBC graphs collate to one batched graph with [B, 3, 3] cell and
    a concatenated shift_idx."""
    from ase import Atoms

    atoms1 = Atoms(
        numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
        pbc=True,
    )
    atoms2 = Atoms(
        numbers=[1, 1, 1],
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
        cell=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        pbc=True,
    )
    g1 = build_graph(atoms1, cutoff=2.5)
    g2 = build_graph(atoms2, cutoff=2.5)

    g = collate_graphs([g1, g2])

    # Batched cell shape
    assert g.cell is not None and g.cell.shape == (2, 3, 3)
    # cell[0] equals g1.cell, cell[1] equals g2.cell
    torch.testing.assert_close(g.cell[0], g1.cell)
    torch.testing.assert_close(g.cell[1], g2.cell)

    # shift_idx is concatenated
    assert g.shift_idx is not None
    assert g.shift_idx.shape[0] == g1.n_edges + g2.n_edges
    assert g.shift_idx.shape[1] == 3

    # batch and edge_index offsets still correct
    assert int(g.batch.max()) == 1
    assert g.batch[: g1.n_atoms].eq(0).all()
    assert g.batch[g1.n_atoms :].eq(1).all()


@pytest.mark.parametrize("model_cls", [InvariantMPNN, EquivariantMPNN])
def test_model_batched_equals_single(ethanol_atoms, model_cls):
    """Per-frame energies from a batched forward must match unbatched forwards.

    This is the structural correctness test for both:
      - collate_graphs offsetting edge_index correctly, and
      - the model's scatter-sum returning the right per-frame split.
    """
    torch.manual_seed(0)
    cutoff = 5.0
    model = model_cls(hidden_dim=16, num_basis=8, cutoff=cutoff, n_layers=2).double()

    graphs = []
    for shift in (0.0, 0.01, -0.01):
        atoms = ethanol_atoms.copy()
        atoms.set_positions(atoms.get_positions() + shift)
        graphs.append(build_graph(atoms, cutoff=cutoff, dtype=torch.float64))

    # Per-frame: 3 scalar energies.
    with torch.no_grad():
        e_singles = torch.stack([model(g) for g in graphs])  # [3]

    # Batched: one [3] tensor.
    batched = collate_graphs(graphs)
    with torch.no_grad():
        e_batched = model(batched)  # [3]

    assert e_batched.shape == (3,)
    assert torch.allclose(e_batched, e_singles, atol=1e-9)
