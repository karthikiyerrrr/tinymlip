"""Unit tests for tinymlip.basis."""

from __future__ import annotations

import torch

from tinymlip.basis import BesselBasis


def test_bessel_basis_shape():
    basis = BesselBasis(num_basis=8, cutoff=5.0)
    r = torch.linspace(0.5, 4.5, 13)
    out = basis(r)
    assert out.shape == (13, 8)
    assert out.dtype == torch.float32


def test_bessel_basis_is_finite_at_very_small_r():
    # b_n(r) = sqrt(2/c) * sin(n*pi*r/c) / r has a removable singularity at r=0.
    # The implementation must stay finite (we clamp r >= 1e-6 in basis.py).
    basis = BesselBasis(num_basis=4, cutoff=5.0)
    r = torch.tensor([1e-8, 1e-6, 1e-3])
    out = basis(r)
    assert torch.isfinite(out).all(), "Bessel basis must not NaN/Inf for tiny r"
