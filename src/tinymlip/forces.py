"""Forces via autograd.

A central architecture rule of this repo: forces are *always* derived from
the energy via autograd, never predicted by a separate head.
This guarantees they are conservative — the line integral of F around any
closed loop in position space is zero, so MD trajectories conserve energy
in NVE ensembles. A separately predicted force field has no such guarantee.
"""

from __future__ import annotations

import torch
from torch import Tensor


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
