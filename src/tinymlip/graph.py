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
