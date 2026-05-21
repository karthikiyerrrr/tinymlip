"""Forces via autograd.

A central architecture rule of this repo: forces are *always* derived from
the energy via autograd, never predicted by a separate head.
This guarantees they are conservative — the line integral of F around any
closed loop in position space is zero, so MD trajectories conserve energy
in NVE ensembles. A separately predicted force field has no such guarantee.
"""

from __future__ import annotations

from dataclasses import replace as _replace

import torch
from torch import Tensor

from tinymlip.graph import AtomGraph


def compute_forces(energy: Tensor, pos: Tensor) -> Tensor:
    """Compute forces as F = -dE/dpos via autograd.

    Args:
        energy: scalar tensor [] from a model forward pass. Must have a
                gradient path back to `pos` (i.e. `pos.requires_grad_(True)`
                was called and `pos` was used in the forward).
        pos:    [N, 3] float tensor — the autograd leaf for atomic positions.

    Returns:
        forces: [N, 3] float — the negative gradient of energy w.r.t. pos.

    `create_graph=True` so notebook 04's force-matching loss can backprop
    through the forces. Inference-only callers pay a small graph-retention
    cost; we keep one path for API simplicity.
    """
    grad = torch.autograd.grad(energy, pos, create_graph=True)[0]
    return -grad


def compute_forces_and_stress(
    model: torch.nn.Module,
    graph: AtomGraph,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Energy, forces (−∂E/∂r), and stress (σ = (1/V) ∂E/∂ε) via autograd.

    Uses the strain-derivative trick (the Knuth/NequIP convention): introduce
    a strain tensor ε, deform positions r → r·(I+ε) and cell c → c·(I+ε),
    forward through the model with ε=0, and read σ from autograd. The layers
    must recompute edge_vec from pos *and* cell inside their forward pass so
    both tensors are in the autograd graph.

    For batched graphs (B frames), uses a per-frame strain [B, 3, 3] so each
    frame's σ is recoverable independently — a single shared strain across the
    batch would only give Σ_b V_b σ_b, useless for training.

    Args:
        model:        model with .forward(graph) -> energy (scalar or [B]).
        graph:        AtomGraph with PBC fields set (cell, shift_idx).
        create_graph: forward the create_graph flag for higher-order grads if
                      the caller needs them (e.g. force-matching loss that
                      backprops through forces).

    Returns:
        E:     scalar Tensor (or [B] when batched).
        F:     [N, 3] forces.
        sigma: [3, 3] symmetric stress (single graph) or [B, 3, 3] (batched).
    """
    if graph.cell is None:
        raise ValueError(
            "compute_forces_and_stress requires a PBC graph (graph.cell must not be None). "
            "Ensure the input was built from atoms with `pbc=True` and a cell."
        )

    # Detach pos/cell from any prior graph and re-attach via fresh leaves.
    pos = graph.pos.detach().clone().requires_grad_(True)
    cell = graph.cell.detach().clone()  # [3, 3] (single) or [B, 3, 3] (batched)

    # Per-frame strain is required for batched stress: one ε per frame so each
    # frame's σ_b = (1/V_b) ∂E_b/∂ε_b is recoverable independently. A single
    # shared strain across the batch would only give Σ_b V_b σ_b — useless.
    is_batched = (graph.batch is not None) and (cell.ndim == 3)
    if is_batched:
        B = int(graph.batch.max().item()) + 1  # noqa: N806 — B is standard for batch size
        strain = torch.zeros(B, 3, 3, dtype=pos.dtype, device=pos.device, requires_grad=True)
        strain_per_atom = strain[graph.batch]  # [N, 3, 3]
        # pos[a] → pos[a] + pos[a] @ strain[batch[a]]
        pos_def = pos + torch.einsum("aj,ajk->ak", pos, strain_per_atom)
        # cell[b] → cell[b] + cell[b] @ strain[b]
        cell_def = cell + torch.einsum("bij,bjk->bik", cell, strain)
    else:
        strain = torch.zeros(3, 3, dtype=pos.dtype, device=pos.device, requires_grad=True)
        pos_def = pos + pos @ strain
        cell_def = cell + cell @ strain

    g_def = _replace(graph, pos=pos_def, cell=cell_def)

    E = model(g_def)  # noqa: N806 — E is standard physics notation for energy; scalar [] or [B]
    E_scalar = E.sum()  # noqa: N806 — scalar gradient handle; per-frame F is still correct
    # because disjoint-union batching has no cross-frame edges.

    grads = torch.autograd.grad(E_scalar, [pos, strain], create_graph=create_graph)
    F = -grads[0]  # noqa: N806 — F is standard physics notation for force; [N, 3]
    if is_batched:
        V = cell.det().abs()  # noqa: N806 — V for volume; [B]
        sigma_raw = grads[1] / V.view(B, 1, 1)  # [B, 3, 3]
        sigma = 0.5 * (sigma_raw + sigma_raw.transpose(-1, -2))
    else:
        V = cell.det().abs()  # noqa: N806 — V for volume; scalar
        sigma_raw = grads[1] / V  # [3, 3]
        sigma = 0.5 * (sigma_raw + sigma_raw.T)

    return E, F, sigma
