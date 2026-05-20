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
