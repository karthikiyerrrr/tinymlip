"""Unit tests for tinymlip.graph."""

from __future__ import annotations

import dataclasses

import numpy as np
import plotly.graph_objects as go
import pytest
import torch
from ase import Atoms

from tinymlip.graph import AtomGraph, _neighbor_list_torch, build_graph
from tinymlip.viz import plot_graph_3d


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


def test_atomgraph_shift_idx_field_defaults_to_none():
    """shift_idx exists, defaults to None, and round-trips when supplied."""
    z = torch.tensor([1, 1], dtype=torch.long)
    pos = torch.zeros((2, 3), dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    edge_dist = torch.tensor([1.0], dtype=torch.float32)

    # Default: None (matches existing non-PBC contract)
    g = AtomGraph(
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        cutoff=5.0,
    )
    assert g.shift_idx is None

    # When supplied, must be [E, 3] long
    shift_idx = torch.tensor([[1, 0, 0]], dtype=torch.long)
    g2 = AtomGraph(
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        cutoff=5.0,
        shift_idx=shift_idx,
    )
    assert g2.shift_idx is not None
    assert g2.shift_idx.dtype == torch.long
    assert g2.shift_idx.shape == (1, 3)


def test_atomgraph_repr_is_short_and_informative():
    # An rMD17-sized molecule: full tensor repr would be hundreds of lines.
    # The custom __repr__ should stay on one line and surface the key fields.
    z = torch.ones(21, dtype=torch.long)
    pos = torch.zeros((21, 3), dtype=torch.float32)
    edge_index = torch.zeros((2, 84), dtype=torch.long)
    edge_vec = torch.zeros((84, 3), dtype=torch.float32)
    edge_dist = torch.zeros((84,), dtype=torch.float32)
    g = AtomGraph(
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        cutoff=5.0,
    )

    r = repr(g)
    assert "\n" not in r, "AtomGraph repr should fit on one line"
    assert len(r) < 120, f"AtomGraph repr should be short, got {len(r)} chars"
    assert "n_atoms=21" in r
    assert "n_edges=84" in r
    assert "cutoff=5.00" in r


def test_neighbor_list_honors_cutoff_and_excludes_self_loops():
    # Atoms on a line at x = 0, 1, 2, 5 (Y=Z=0).
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    edge_index, edge_vec, edge_dist = _neighbor_list_torch(pos, cutoff=1.5)

    # Within 1.5 Å: (0,1), (1,0), (1,2), (2,1). Nothing touching atom 3.
    edges = {tuple(e) for e in edge_index.t().tolist()}
    assert edges == {(0, 1), (1, 0), (1, 2), (2, 1)}

    # No self-loops.
    for i, j in edges:
        assert i != j

    # edge_vec[k] = pos[dst] - pos[src]; ||edge_vec|| matches edge_dist.
    assert torch.allclose(edge_dist, edge_vec.norm(dim=-1))


def test_neighbor_list_widens_with_cutoff():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    edge_index, _, _ = _neighbor_list_torch(pos, cutoff=2.5)
    edges = {tuple(e) for e in edge_index.t().tolist()}

    # 2.5 Å now includes (0,2) and (2,0); atom 3 is still 3 Å away from
    # the nearest neighbor and stays isolated.
    assert (0, 2) in edges and (2, 0) in edges
    assert all(3 not in pair for pair in edges)


def test_build_graph_returns_documented_shapes_and_dtypes():
    # 4-atom toy: H, C, C, H along x.
    atoms = Atoms(
        numbers=[1, 6, 6, 1],
        positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.6, 0.0, 0.0], [3.7, 0.0, 0.0]],
    )

    g = build_graph(atoms, cutoff=2.0)

    assert g.z.dtype == torch.long
    assert g.pos.dtype == torch.float32
    assert g.edge_index.dtype == torch.long
    assert g.edge_vec.dtype == torch.float32
    assert g.edge_dist.dtype == torch.float32

    assert g.z.shape == (4,)
    assert g.pos.shape == (4, 3)
    assert g.edge_index.shape[0] == 2
    e = g.edge_index.shape[1]
    assert g.edge_vec.shape == (e, 3)
    assert g.edge_dist.shape == (e,)

    assert g.cutoff == 2.0
    assert g.cell is None
    assert g.pbc == (False, False, False)


def test_graph_is_translation_invariant():
    atoms = Atoms(
        numbers=[1, 6, 6, 8],
        positions=[[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [2.4, 0.1, 0.3], [3.5, 0.0, 0.0]],
    )
    g1 = build_graph(atoms, cutoff=2.0)

    shifted = atoms.copy()
    shifted.translate([10.0, -3.0, 7.0])
    g2 = build_graph(shifted, cutoff=2.0)

    e1 = {tuple(e) for e in g1.edge_index.t().tolist()}
    e2 = {tuple(e) for e in g2.edge_index.t().tolist()}
    assert e1 == e2

    d1, _ = g1.edge_dist.sort()
    d2, _ = g2.edge_dist.sort()
    assert torch.allclose(d1, d2)


def test_graph_relabels_under_permutation():
    atoms = Atoms(
        numbers=[1, 6, 6, 8],
        positions=[[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [2.4, 0.1, 0.3], [3.5, 0.0, 0.0]],
    )
    g1 = build_graph(atoms, cutoff=2.0)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(atoms))  # array, e.g. [2, 0, 3, 1]
    inverse = np.argsort(perm)

    shuffled = Atoms(
        numbers=atoms.numbers[perm],
        positions=atoms.positions[perm],
    )
    g2 = build_graph(shuffled, cutoff=2.0)

    # For every (i, j) in g1, the edge (inverse-relabel(i), inverse-relabel(j))
    # should exist in g2 because shuffled[k] == original[perm[k]],
    # so original index `i` maps to shuffled index `inverse[i]`.
    e1 = {tuple(e) for e in g1.edge_index.t().tolist()}
    e2 = {tuple(e) for e in g2.edge_index.t().tolist()}
    expected = {(int(inverse[i]), int(inverse[j])) for (i, j) in e1}
    assert expected == e2


def test_build_graph_pbc_returns_cell_and_shift_idx():
    """Periodic systems now return cell and shift_idx (not NotImplementedError)."""
    atoms = Atoms(
        numbers=[6, 6],
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=[True, True, True],
    )

    g = build_graph(atoms, cutoff=5.0)

    # PBC path must populate cell and shift_idx
    assert g.cell is not None
    assert g.cell.shape == (3, 3)
    assert g.shift_idx is not None
    assert g.shift_idx.shape[0] == g.n_edges
    assert g.shift_idx.shape[1] == 3
    assert g.pbc == (True, True, True)


def test_element_color_falls_back_for_unknown_z():
    from tinymlip.viz import element_color

    assert element_color(1) == "#ffffff"  # H is white in CPK-ish palette
    assert element_color(6) == "#444444"  # C is dark grey
    assert element_color(8) == "#ff0d0d"  # O is red
    # Unknown / exotic element — should fall back, not raise.
    assert element_color(118) == "#888888"


def test_element_radius_uses_ase_covalent_radii():
    from ase.data import covalent_radii

    from tinymlip.viz import element_radius

    # We return ASE's covalent radius for the element, in Å.
    assert element_radius(1) == pytest.approx(covalent_radii[1])
    assert element_radius(6) == pytest.approx(covalent_radii[6])


def test_plot_graph_3d_returns_a_plotly_figure():
    atoms = Atoms(
        numbers=[1, 6, 6, 8],
        positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.4, 0.0, 0.0], [3.5, 0.0, 0.0]],
    )
    g = build_graph(atoms, cutoff=2.0)

    fig = plot_graph_3d(g)

    assert isinstance(fig, go.Figure)
    # At least: one atom-scatter trace, plus at least one of (bond lines, edge lines).
    assert len(fig.data) >= 2

    # Atom trace must always be present regardless of show_* flags.
    fig_atoms_only = plot_graph_3d(g, show_bonds=False, show_edges=False)
    assert len(fig_atoms_only.data) == 1
    assert fig_atoms_only.data[0].name == "atoms"


def test_plot_edge_distance_histogram_returns_a_plotly_figure():
    from tinymlip.viz import plot_edge_distance_histogram

    atoms = Atoms(
        numbers=[1, 6, 6, 8],
        positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.4, 0.0, 0.0], [3.5, 0.0, 0.0]],
    )
    g = build_graph(atoms, cutoff=2.0)

    fig = plot_edge_distance_histogram(g)

    assert isinstance(fig, go.Figure)
    # Two histogram traces (kept, excluded).
    assert len(fig.data) >= 1
    # A vertical cutoff line is drawn as a layout shape (plotly Shape objects
    # expose `.type`, not a dict-style .get()).
    shapes = fig.layout.shapes or ()
    assert any(getattr(s, "type", None) == "line" for s in shapes)


def test_graph_stats_md_contains_size_and_degree_numbers():
    from tinymlip.viz import graph_stats_md

    atoms = Atoms(
        numbers=[1, 6, 6, 8],
        positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.4, 0.0, 0.0], [3.5, 0.0, 0.0]],
    )
    g = build_graph(atoms, cutoff=2.0)

    text = graph_stats_md(g)

    assert isinstance(text, str)
    assert "|V|" in text
    assert "|E|" in text
    assert str(g.n_atoms) in text
    assert str(g.n_edges) in text
    assert "mean deg" in text


def test_atomgraph_batch_field_defaults_none(ethanol_atoms):
    """`batch` is optional. nb 03 path: a single-graph build must leave it None."""
    from tinymlip.graph import build_graph

    graph = build_graph(ethanol_atoms, cutoff=5.0)
    assert graph.batch is None


def test_build_graph_pbc_simple_cubic_two_atoms():
    """Two atoms in a cubic cell — image enumeration is hand-checkable.

    Cell: 4 Å cube. Atoms at (0,0,0) and (2,0,0). Cutoff 2.5 Å.

    Within the central cell: 0<->1 at distance 2.0 (1 image-pair, both directions).
    Plus the periodic image of atom 1 at (-2, 0, 0) is 2.0 from atom 0 (shift S=(-1,0,0)),
    and the image of atom 0 at (4, 0, 0) is 2.0 from atom 1 (shift S=(+1,0,0)).
    No other shifts are within cutoff (next-nearest is 4 Å along y or z).

    Total edges: 4 (two same-cell + two image-pair, both directions).
    """
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
        pbc=True,
    )
    g = build_graph(atoms, cutoff=2.5)

    assert g.n_atoms == 2
    assert g.shift_idx is not None
    # Exactly 4 edges, each at distance 2.0
    assert g.n_edges == 4
    torch.testing.assert_close(g.edge_dist, torch.full((4,), 2.0), atol=1e-5, rtol=0)
    # Each edge has |shift|_inf in {0, 1}
    assert g.shift_idx.abs().max().item() == 1
    # cell and pbc round-tripped onto the graph
    assert g.cell is not None and g.cell.shape == (3, 3)
    assert g.pbc == (True, True, True)
