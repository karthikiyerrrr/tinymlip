"""Unit tests for tinymlip.graph."""

from __future__ import annotations

import torch

from tinymlip.graph import AtomGraph


def test_atomgraph_holds_documented_fields():
    z = torch.tensor([1, 6, 6, 1], dtype=torch.long)
    pos = torch.zeros((4, 3), dtype=torch.float32)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_vec = torch.zeros((0, 3), dtype=torch.float32)
    edge_dist = torch.zeros((0,), dtype=torch.float32)

    g = AtomGraph(
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        cutoff=5.0,
    )

    assert g.n_atoms == 4
    assert g.n_edges == 0
    assert g.cell is None
    assert g.pbc == (False, False, False)
