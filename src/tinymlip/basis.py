"""Radial basis functions for message-passing layers.

Two small `nn.Module`s: a Bessel radial basis and a cosine cutoff envelope.
They are kept separate (rather than fused into a single `RadialFeaturizer`)
so notebook 02 can plot each on its own axis — the visual separation is
itself a teaching point.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class BesselBasis(nn.Module):
    """Bessel radial basis on [0, cutoff].

    b_n(r) = sqrt(2/cutoff) * sin(n*pi*r/cutoff) / r  for n = 1..num_basis.

    Popularized for MLIPs by Klicpera et al. 2020 (DimeNet). The basis is
    orthogonal on [0, cutoff], so the model needs fewer basis functions to
    cover the radial range than a Gaussian RBF would.

    The 1/r factor is finite for r > 0 (sin(0)=0 cancels it) but numerically
    unstable for very small r. We clamp r >= 1e-6; neighbor lists exclude
    self-loops, so real edges are always > 0 in practice.
    """

    def __init__(self, num_basis: int, cutoff: float) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = float(cutoff)
        # frequencies n*pi/cutoff for n = 1..num_basis; registered as a buffer
        # so it moves with .to(device).
        n = torch.arange(1, num_basis + 1, dtype=torch.float32)
        self.register_buffer("freqs", n * math.pi / self.cutoff)
        self.prefactor: float = math.sqrt(2.0 / self.cutoff)

    def forward(self, r: Tensor) -> Tensor:
        # r: [E] -> [E, num_basis]
        r_safe = r.clamp(min=1e-6).unsqueeze(-1)  # [E, 1]
        return self.prefactor * torch.sin(self.freqs * r_safe) / r_safe


class CosineEnvelope(nn.Module):
    """Smooth cutoff envelope: 1 at r=0, exactly 0 at r=cutoff.

    f_cut(r) = 0.5 * (cos(pi*r/cutoff) + 1) for r <= cutoff else 0.

    Multiplying the radial basis by this envelope guarantees that any
    per-edge quantity vanishes as r -> cutoff. That keeps the predicted
    energy continuous as atoms cross the cutoff boundary, which in turn
    keeps forces (= -dE/dr) well-defined.
    """

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, r: Tensor) -> Tensor:
        # r: [E] -> [E]
        inside = 0.5 * (torch.cos(math.pi * r / self.cutoff) + 1.0)
        return torch.where(r <= self.cutoff, inside, torch.zeros_like(r))
