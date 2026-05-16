"""Unit tests for tinymlip.graph."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tinymlip.graph import AtomGraph


def test_atomgraph_holds_documented_fields():
    z = torch.tensor([1, 6, 6, 1], dtype=torch.long)
    pos = torch.zeros((4, 3), dtype=torch.float32)
    # Two edges so `n_edges` is exercised on a non-trivial value
    # (the zero-edge case can't distinguish edge_index.shape[1] from shape[0]).
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_vec = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32)
    edge_dist = torch.tensor([1.0, 1.0], dtype=torch.float32)

    g = AtomGraph(
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        cutoff=5.0,
    )

    # Shape / property contract
    assert g.n_atoms == 4
    assert g.n_edges == 2

    # Dtype contract — layers downstream rely on these
    assert g.z.dtype == torch.long
    assert g.pos.dtype == torch.float32
    assert g.edge_index.dtype == torch.long
    assert g.edge_vec.dtype == torch.float32
    assert g.edge_dist.dtype == torch.float32

    # Defaults
    assert g.cell is None
    assert g.pbc == (False, False, False)

    # frozen=True: mutating a field must raise
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.cutoff = 3.0  # type: ignore[misc]
