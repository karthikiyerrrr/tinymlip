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
        self.register_buffer(
            "prefactor", torch.tensor(math.sqrt(2.0 / self.cutoff), dtype=torch.float32)
        )

    def forward(self, r: Tensor) -> Tensor:
        # r: [E] -> [E, num_basis]
        r_safe = r.clamp(min=1e-6).unsqueeze(-1)  # [E, 1]
        return self.prefactor * torch.sin(self.freqs * r_safe) / r_safe
