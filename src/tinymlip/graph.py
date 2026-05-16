"""Build radial-cutoff graphs from ASE Atoms objects.

A graph here is the standard atomistic-ML object: nodes are atoms, edges
connect any pair of atoms within `cutoff` Angstroms. We follow field
convention: edges are directed both ways (every pair contributes (i,j) and
(j,i)), self-loops are excluded, and `edge_vec = pos[dst] - pos[src]`.

For non-periodic systems we use a hand-written O(N^2) cdist scan — it's
five conceptual lines and rMD17 molecules have <= 21 atoms. The PBC path
is signaled in the signature but not implemented; it lands in notebook 06.
"""

from __future__ import annotations

from dataclasses import dataclass

import ase
import torch


@dataclass(frozen=True)
class AtomGraph:
    """A molecular graph: nodes are atoms, edges are pairs within a cutoff radius.

    Field shapes are part of the contract. All tensors live on a single device.

    Edge geometry (`edge_vec`, `edge_dist`) is cached at construction time so
    layers can read it directly. Models that need autograd-correct forces must
    recompute `edge_vec` from `pos` inside their forward pass — `pos` must be
    the autograd leaf, not `edge_vec`. See models.py when it lands.
    """

    z: torch.Tensor  # [N]    long       — atomic numbers
    pos: torch.Tensor  # [N, 3] float      — Cartesian positions (Å)
    edge_index: torch.Tensor  # [2, E] long       — (src, dst), directed
    edge_vec: torch.Tensor  # [E, 3] float      — pos[dst] - pos[src]
    edge_dist: torch.Tensor  # [E]    float      — ||edge_vec||
    cutoff: float
    cell: torch.Tensor | None = None  # [3, 3] float; None for non-PBC
    pbc: tuple[bool, bool, bool] = (False, False, False)

    @property
    def n_atoms(self) -> int:
        return int(self.z.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def _neighbor_list_torch(
    pos: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairs within `cutoff`, both directions, no self-loops.

    Hand-written torch-only neighbor list. Computed via pairwise distance
    matrix; O(N²) but negligible for small molecules (N ≤ 21 on rMD17).

    Args:
        pos: [N, 3] float — atomic positions in Angstroms.
        cutoff: float — distance cutoff in Angstroms.

    Returns:
        edge_index: [2, E] long — (src, dst) pairs, directed both ways.
        edge_vec: [E, 3] float — pos[dst] - pos[src].
        edge_dist: [E] float — Euclidean distance for each edge.
    """
    # pos.unsqueeze(0): [1, N, 3] ; pos.unsqueeze(1): [N, 1, 3].
    # diff[i, j, :] = pos[j] - pos[i], so diff[src, dst] = pos[dst] - pos[src].
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, 3]
    dist = diff.norm(dim=-1)  # [N, N]
    mask = (dist > 0) & (dist <= cutoff)  # excludes the diagonal (i == j)
    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst])  # [2, E]
    edge_vec = diff[src, dst]  # [E, 3]
    edge_dist = dist[src, dst]  # [E]
    return edge_index, edge_vec, edge_dist


def build_graph(
    atoms: ase.Atoms,
    *,
    cutoff: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> AtomGraph:
    """Build a radial-cutoff graph from an ASE Atoms object.

    Non-periodic systems use a hand-written O(N^2) pair scan (see
    `_neighbor_list_torch`). rMD17 molecules have <= 21 atoms; the cost is
    dominated by Python overhead and we prefer the readable implementation.

    For periodic systems (any of `atoms.pbc` is True), raises
    NotImplementedError. PBC support arrives with notebook 06.

    Args:
        atoms:  ASE Atoms; `atoms.numbers` and `atoms.positions` are read.
        cutoff: radial cutoff in Å. Edges connect pairs with 0 < dist <= cutoff.
        dtype:  floating dtype for positions and edge features.
        device: device for all returned tensors.
    """
    if any(atoms.pbc):
        raise NotImplementedError("PBC support added in notebook 06")

    z = torch.as_tensor(atoms.numbers, dtype=torch.long, device=device)
    pos = torch.as_tensor(atoms.positions, dtype=dtype, device=device)
    edge_index, edge_vec, edge_dist = _neighbor_list_torch(pos, cutoff)

    return AtomGraph(
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        cutoff=float(cutoff),
        cell=None,
        pbc=(False, False, False),
    )
