"""Unit tests for tinymlip.layers."""

from __future__ import annotations

import numpy as np
import torch

from tinymlip.graph import AtomGraph
from tinymlip.layers import AtomicReadout, EquivariantInteraction, InvariantInteraction


def _random_graph(n_atoms: int = 9, cutoff: float = 2.5, seed: int = 0) -> AtomGraph:
    """Small random graph fixture; positions in a 6-Angstrom cube."""
    rng = np.random.default_rng(seed)
    pos = torch.tensor(rng.uniform(-3.0, 3.0, size=(n_atoms, 3)), dtype=torch.float32)
    z = torch.tensor(rng.integers(1, 10, size=(n_atoms,)), dtype=torch.long)
    # Build edges by O(N^2) cutoff scan (mirrors graph._neighbor_list_torch).
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    dist = diff.norm(dim=-1)
    mask = (dist > 0) & (dist <= cutoff)
    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst])
    edge_vec = diff[src, dst]
    edge_dist = dist[src, dst]
    return AtomGraph(
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        cutoff=cutoff,
    )


def _random_rotation(seed: int = 0) -> torch.Tensor:
    """A random SO(3) matrix via QR decomposition of a Gaussian matrix."""
    rng = np.random.default_rng(seed)
    a = torch.tensor(rng.standard_normal(size=(3, 3)), dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    # Make sure det(q) = +1 (proper rotation, not a reflection)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _rotate_graph(graph: AtomGraph, R: torch.Tensor) -> AtomGraph:  # noqa: N803 — R is standard rotation matrix notation
    """Apply rotation R to positions; recompute edge_vec/edge_dist."""
    pos = graph.pos @ R.T
    src, dst = graph.edge_index
    edge_vec = pos[dst] - pos[src]
    edge_dist = edge_vec.norm(dim=-1)
    return AtomGraph(
        z=graph.z,
        pos=pos,
        edge_index=graph.edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        cutoff=graph.cutoff,
    )


def test_invariant_interaction_shape():
    torch.manual_seed(0)
    graph = _random_graph()
    layer = InvariantInteraction(hidden_dim=16, num_basis=8, cutoff=graph.cutoff)
    x = torch.randn(graph.n_atoms, 16)
    out = layer(x, graph)
    assert out.shape == (graph.n_atoms, 16)


def test_invariant_interaction_permutation_equivariance():
    torch.manual_seed(0)
    graph = _random_graph()
    layer = InvariantInteraction(hidden_dim=16, num_basis=8, cutoff=graph.cutoff)
    x = torch.randn(graph.n_atoms, 16)
    out_ref = layer(x, graph)

    # Shuffle atoms; relabel edge_index accordingly.
    rng = np.random.default_rng(1)
    perm = torch.tensor(rng.permutation(graph.n_atoms), dtype=torch.long)
    inverse = torch.argsort(perm)
    pos_p = graph.pos[perm]
    z_p = graph.z[perm]
    src, dst = graph.edge_index
    edge_index_p = torch.stack([inverse[src], inverse[dst]])
    edge_vec_p = pos_p[edge_index_p[1]] - pos_p[edge_index_p[0]]
    edge_dist_p = edge_vec_p.norm(dim=-1)
    graph_p = AtomGraph(
        z=z_p,
        pos=pos_p,
        edge_index=edge_index_p,
        edge_vec=edge_vec_p,
        edge_dist=edge_dist_p,
        cutoff=graph.cutoff,
    )
    x_p = x[perm]
    out_p = layer(x_p, graph_p)

    # Output should permute the same way as the input.
    assert torch.allclose(out_p, out_ref[perm], atol=1e-5)


def test_invariant_interaction_translation_invariance():
    torch.manual_seed(0)
    graph = _random_graph()
    layer = InvariantInteraction(hidden_dim=16, num_basis=8, cutoff=graph.cutoff)
    x = torch.randn(graph.n_atoms, 16)
    out_ref = layer(x, graph)

    shift = torch.tensor([10.0, -3.0, 7.0])
    pos_t = graph.pos + shift
    # edge_vec is unchanged by a constant shift, so we can reuse it
    graph_t = AtomGraph(
        z=graph.z,
        pos=pos_t,
        edge_index=graph.edge_index,
        edge_vec=graph.edge_vec,
        edge_dist=graph.edge_dist,
        cutoff=graph.cutoff,
    )
    out_t = layer(x, graph_t)
    assert torch.allclose(out_t, out_ref, atol=1e-6)


def test_invariant_interaction_rotation_invariance():
    torch.manual_seed(0)
    graph = _random_graph()
    layer = InvariantInteraction(hidden_dim=16, num_basis=8, cutoff=graph.cutoff)
    x = torch.randn(graph.n_atoms, 16)
    out_ref = layer(x, graph)

    R = _random_rotation(seed=2)  # noqa: N806 — R is standard rotation matrix notation
    graph_r = _rotate_graph(graph, R)
    out_r = layer(x, graph_r)
    # Invariant: output unchanged under rotation of inputs.
    assert torch.allclose(out_r, out_ref, atol=1e-5)


def test_equivariant_interaction_shape():
    torch.manual_seed(0)
    graph = _random_graph()
    layer = EquivariantInteraction(hidden_dim=16, num_basis=8, cutoff=graph.cutoff)
    s = torch.randn(graph.n_atoms, 16)
    v = torch.zeros(graph.n_atoms, 16, 3)
    s_out, v_out = layer(s, v, graph)
    assert s_out.shape == (graph.n_atoms, 16)
    assert v_out.shape == (graph.n_atoms, 16, 3)


def test_equivariant_interaction_permutation_equivariance():
    torch.manual_seed(0)
    graph = _random_graph()
    layer = EquivariantInteraction(hidden_dim=16, num_basis=8, cutoff=graph.cutoff)
    s = torch.randn(graph.n_atoms, 16)
    v = torch.randn(graph.n_atoms, 16, 3)
    s_ref, v_ref = layer(s, v, graph)

    rng = np.random.default_rng(1)
    perm = torch.tensor(rng.permutation(graph.n_atoms), dtype=torch.long)
    inverse = torch.argsort(perm)
    pos_p = graph.pos[perm]
    z_p = graph.z[perm]
    src, dst = graph.edge_index
    edge_index_p = torch.stack([inverse[src], inverse[dst]])
    edge_vec_p = pos_p[edge_index_p[1]] - pos_p[edge_index_p[0]]
    edge_dist_p = edge_vec_p.norm(dim=-1)
    graph_p = AtomGraph(
        z=z_p,
        pos=pos_p,
        edge_index=edge_index_p,
        edge_vec=edge_vec_p,
        edge_dist=edge_dist_p,
        cutoff=graph.cutoff,
    )
    s_out, v_out = layer(s[perm], v[perm], graph_p)
    assert torch.allclose(s_out, s_ref[perm], atol=1e-5)
    assert torch.allclose(v_out, v_ref[perm], atol=1e-5)


def test_equivariant_interaction_translation_invariance():
    torch.manual_seed(0)
    graph = _random_graph()
    layer = EquivariantInteraction(hidden_dim=16, num_basis=8, cutoff=graph.cutoff)
    s = torch.randn(graph.n_atoms, 16)
    v = torch.randn(graph.n_atoms, 16, 3)
    s_ref, v_ref = layer(s, v, graph)

    shift = torch.tensor([10.0, -3.0, 7.0])
    pos_t = graph.pos + shift
    graph_t = AtomGraph(
        z=graph.z,
        pos=pos_t,
        edge_index=graph.edge_index,
        edge_vec=graph.edge_vec,
        edge_dist=graph.edge_dist,
        cutoff=graph.cutoff,
    )
    s_out, v_out = layer(s, v, graph_t)
    assert torch.allclose(s_out, s_ref, atol=1e-6)
    assert torch.allclose(v_out, v_ref, atol=1e-6)


def test_equivariant_interaction_rotation_equivariance():
    # IMPORTANT: input v must be non-zero. With v=0, the propagation message
    # and the update-phase mixing both no-op, so a bug in those code paths
    # would not be caught.
    torch.manual_seed(0)
    graph = _random_graph()
    layer = EquivariantInteraction(hidden_dim=16, num_basis=8, cutoff=graph.cutoff)
    s = torch.randn(graph.n_atoms, 16)
    v = torch.randn(graph.n_atoms, 16, 3)
    s_ref, v_ref = layer(s, v, graph)

    R = _random_rotation(seed=3)  # noqa: N806 — R is standard rotation matrix notation
    graph_r = _rotate_graph(graph, R)
    # Vectors rotate with the molecule.
    v_rot = v @ R.T  # [N, F, 3] -> rotate last dim by R^T
    s_out, v_out = layer(s, v_rot, graph_r)

    # Scalars unchanged; vectors rotated by R.
    assert torch.allclose(s_out, s_ref, atol=1e-4)
    assert torch.allclose(v_out, v_ref @ R.T, atol=1e-4)


def test_atomic_readout_shape_and_independence():
    """Readout produces [N, 1] from [N, F] and is per-atom (no cross-talk)."""
    torch.manual_seed(0)
    readout = AtomicReadout(hidden_dim=16)
    x = torch.randn(5, 16)

    y = readout(x)
    assert y.shape == (5, 1)

    # Per-atom independence: changing atom 0's features must not change atom 3's output.
    x2 = x.clone()
    x2[0] = torch.randn(16)
    y2 = readout(x2)
    assert torch.allclose(y[3], y2[3])
    assert not torch.allclose(y[0], y2[0])
