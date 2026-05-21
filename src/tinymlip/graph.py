"""Build radial-cutoff graphs from ASE Atoms objects.

A graph here is the standard atomistic-ML object: nodes are atoms, edges
connect any pair of atoms within `cutoff` Angstroms. We follow field
convention: edges are directed both ways (every pair contributes (i,j) and
(j,i)), self-loops are excluded, and `edge_vec = pos[dst] - pos[src]`.

For non-periodic systems we use a hand-written O(N^2) cdist scan — it's
five conceptual lines and rMD17 molecules have <= 21 atoms. For PBC, we
delegate to ASE's primitive_neighbor_list which handles arbitrary cell shapes
(orthorhombic, triclinic, high aspect ratio) and reports integer lattice
shifts per edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import ase
import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list


@dataclass(frozen=True)
class AtomGraph:
    """A molecular graph: nodes are atoms, edges are pairs within a cutoff radius.

    Field shapes are part of the contract. All tensors live on a single device.

    Edge geometry (`edge_vec`, `edge_dist`) is cached at construction time so
    layers can read it directly. Models that need autograd-correct forces must
    recompute `edge_vec` from `pos` inside their forward pass — `pos` must be
    the autograd leaf, not `edge_vec`. See models.py when it lands. Under PBC,
    `shift_idx` carries the integer lattice offset `S` for each edge so the model
    can recompute `edge_vec = pos[dst] - pos[src] + S @ cell` inside its forward
    pass. `None` on the non-PBC path.

    Batching: `collate_graphs([g1, g2, ...])` builds the disjoint union of
    several AtomGraphs into one, populating `batch` so per-frame readouts can
    scatter-sum correctly. Used by notebook 04 onwards.
    """

    z: torch.Tensor  # [N]    long       — atomic numbers
    pos: torch.Tensor  # [N, 3] float      — Cartesian positions (Å)
    edge_index: torch.Tensor  # [2, E] long       — (src, dst), directed
    edge_vec: torch.Tensor  # [E, 3] float      — pos[dst] - pos[src]
    edge_dist: torch.Tensor  # [E]    float      — ||edge_vec||
    cutoff: float
    cell: torch.Tensor | None = None  # [3, 3] float; None for non-PBC
    pbc: tuple[bool, bool, bool] = (False, False, False)
    batch: torch.Tensor | None = None  # [N] long — frame id per atom; None for a single graph
    shift_idx: torch.Tensor | None = (
        None  # [E, 3] long — integer lattice shifts per edge; None for non-PBC
    )

    @property
    def n_atoms(self) -> int:
        return int(self.z.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def __repr__(self) -> str:
        # The dataclass-generated repr dumps every tensor — wall of numbers in a
        # notebook cell. Print a one-line summary instead; fields stay accessible.
        if self.batch is None:
            return (
                f"AtomGraph(n_atoms={self.n_atoms}, n_edges={self.n_edges}, "
                f"cutoff={self.cutoff:.2f}, pbc={any(self.pbc)})"
            )
        n_frames = int(self.batch.max()) + 1
        return (
            f"AtomGraph(n_frames={n_frames}, n_atoms={self.n_atoms}, "
            f"n_edges={self.n_edges}, cutoff={self.cutoff:.2f}, pbc={any(self.pbc)})"
        )


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


def _neighbor_list_ase(
    atoms: ase.Atoms,
    cutoff: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Periodic neighbor list via ASE.

    Delegates to `ase.neighborlist.primitive_neighbor_list`, the same backend
    SchNetPack uses. Returns directed edges (both (i, j) and (j, i)), excludes
    self-loops, and reports lattice shifts so the model can recompute
    `edge_vec = pos[j] - pos[i] + S @ cell` inside its forward pass — this is
    what lets autograd flow through both positions and cell (the latter is
    needed for stress via the strain trick).

    Args:
        atoms:  ASE Atoms with `pbc` and `cell` set.
        cutoff: radial cutoff in Å.
        dtype:  floating dtype for positions and edge_vec.
        device: torch device for returned tensors.

    Returns:
        edge_index: [2, E] long — (src, dst), directed both ways, no self-loops.
        edge_vec:   [E, 3] float — pos[dst] - pos[src] (already unwrapped through PBC).
        edge_dist:  [E] float — Euclidean distance for each edge.
        shift_idx:  [E, 3] long — integer lattice shifts S, one row per edge.
    """
    # quantities "ijDdS": i,j atom indices, D unwrapped vector, d distance, S shift (integers).
    # self_interaction=False excludes (i, i, S=0). image-of-self at S != 0 is included
    # when within cutoff (correct: an atom DOES interact with its own periodic image).
    i, j, D, d, S = primitive_neighbor_list(  # noqa: N806
        "ijDdS",
        pbc=atoms.pbc,
        cell=atoms.cell.array,
        positions=atoms.positions,
        cutoff=cutoff,
        self_interaction=False,
    )
    edge_index = torch.as_tensor(
        np.stack([i, j], axis=0), dtype=torch.long, device=device
    )  # [2, E]
    edge_vec = torch.as_tensor(D, dtype=dtype, device=device)  # [E, 3]
    edge_dist = torch.as_tensor(d, dtype=dtype, device=device)  # [E]
    shift_idx = torch.as_tensor(S, dtype=torch.long, device=device)  # [E, 3]
    return edge_index, edge_vec, edge_dist, shift_idx


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

    Periodic systems delegate to `ase.neighborlist.primitive_neighbor_list`
    via `_neighbor_list_ase` — the same backend SchNetPack uses. The returned
    graph carries `shift_idx` so the model can recompute periodic edge_vecs
    inside its forward pass and propagate autograd through the cell (needed
    for stress).

    Args:
        atoms:  ASE Atoms; `atoms.numbers`, `atoms.positions`, and (for PBC)
                `atoms.cell` and `atoms.pbc` are read.
        cutoff: radial cutoff in Å. Edges connect pairs with 0 < dist <= cutoff.
        dtype:  floating dtype for positions and edge features.
        device: device for all returned tensors.
    """
    z = torch.as_tensor(atoms.numbers, dtype=torch.long, device=device)
    pos = torch.as_tensor(atoms.positions, dtype=dtype, device=device)

    if any(atoms.pbc):
        edge_index, edge_vec, edge_dist, shift_idx = _neighbor_list_ase(
            atoms, cutoff, dtype, device
        )
        cell = torch.as_tensor(atoms.cell.array, dtype=dtype, device=device)
        pbc = tuple(bool(b) for b in atoms.pbc)  # tuple[bool, bool, bool]
        return AtomGraph(
            z=z,
            pos=pos,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_dist=edge_dist,
            cutoff=float(cutoff),
            cell=cell,
            pbc=pbc,
            shift_idx=shift_idx,
        )

    # Non-PBC path: unchanged
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


def collate_graphs(graphs: list[AtomGraph]) -> AtomGraph:
    """Disjoint-union a list of AtomGraphs into one batched AtomGraph.

    Standard MLIP batching trick: stack the per-frame graphs into one large
    graph with the union of nodes and edges, and add a `batch` vector that
    records which frame each atom belongs to. Each graph's edges only connect
    its own atoms, so a single forward pass through any layer handles all
    frames at once.

    Under PBC, each frame may have a different cell — these are stacked into a
    [B, 3, 3] tensor. Each edge's `shift_idx` (lattice offset) is concatenated.
    The model selects the right per-edge cell via `cell[batch[edge_index[0]]]`.

    Args:
        graphs: list of AtomGraph (single-frame, batch is None). Must share
            `cutoff`. All graphs in a batch must be uniformly PBC or uniformly
            non-PBC — mixing is rejected.

    Returns:
        A new AtomGraph with:
          - concatenated `z`, `pos`, `edge_vec`, `edge_dist`, `shift_idx` (when PBC);
          - `edge_index` offset per-frame so all indices live in [0, sum_N);
          - `cell` stacked to [B, 3, 3] (when PBC), else None;
          - `batch` tensor of shape [sum_N] mapping atom -> frame.
    """
    if not graphs:
        raise ValueError("collate_graphs requires at least one graph")
    cutoff = graphs[0].cutoff
    if any(g.cutoff != cutoff for g in graphs):
        raise ValueError("collate_graphs: all graphs must share the same cutoff")
    pbc_flags = [any(g.pbc) for g in graphs]
    if any(pbc_flags) and not all(pbc_flags):
        raise ValueError("collate_graphs: cannot mix PBC and non-PBC graphs in one batch")
    is_pbc = all(pbc_flags)

    z = torch.cat([g.z for g in graphs], dim=0)  # [sum_N]
    pos = torch.cat([g.pos for g in graphs], dim=0)  # [sum_N, 3]
    edge_vec = torch.cat([g.edge_vec for g in graphs], dim=0)  # [sum_E, 3]
    edge_dist = torch.cat([g.edge_dist for g in graphs], dim=0)  # [sum_E]

    edge_indices: list[torch.Tensor] = []
    batch_pieces: list[torch.Tensor] = []
    offset = 0
    for frame_id, g in enumerate(graphs):
        edge_indices.append(g.edge_index + offset)  # [2, E_k]
        batch_pieces.append(torch.full((g.n_atoms,), frame_id, dtype=torch.long, device=g.z.device))
        offset += g.n_atoms
    edge_index = torch.cat(edge_indices, dim=1)  # [2, sum_E]
    batch = torch.cat(batch_pieces, dim=0)  # [sum_N]

    if is_pbc:
        shift_idx = torch.cat([g.shift_idx for g in graphs], dim=0)  # [sum_E, 3]
        cell = torch.stack([g.cell for g in graphs], dim=0)  # [B, 3, 3]
        pbc_out = graphs[0].pbc  # uniform across the batch
    else:
        shift_idx = None
        cell = None
        pbc_out = (False, False, False)

    return AtomGraph(
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        cutoff=cutoff,
        cell=cell,
        pbc=pbc_out,
        batch=batch,
        shift_idx=shift_idx,
    )
