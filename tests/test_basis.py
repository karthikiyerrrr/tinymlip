"""Unit tests for tinymlip.basis."""

from __future__ import annotations

import torch

from tinymlip.basis import BesselBasis, CosineEnvelope


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


def test_cosine_envelope_boundary_values():
    env = CosineEnvelope(cutoff=5.0)
    r = torch.tensor([0.0, 5.0, 5.1, 10.0])
    out = env(r)
    assert torch.allclose(out[0], torch.tensor(1.0))
    assert torch.allclose(out[1], torch.tensor(0.0), atol=1e-7)
    # r > cutoff must return exactly zero (guards the torch.where condition).
    assert out[2].item() == 0.0
    assert out[3].item() == 0.0


def test_cosine_envelope_is_monotone_non_increasing():
    env = CosineEnvelope(cutoff=5.0)
    r = torch.linspace(0.0, 5.0, 50)
    out = env(r)
    diffs = out[1:] - out[:-1]
    assert (diffs <= 1e-7).all(), "envelope must be non-increasing on [0, cutoff]"


def test_basis_times_envelope_is_zero_at_cutoff():
    # Multiplying basis by envelope must vanish exactly at r = cutoff
    # so per-edge quantities are continuous as atoms cross the boundary.
    basis = BesselBasis(num_basis=8, cutoff=5.0)
    env = CosineEnvelope(cutoff=5.0)
    r = torch.tensor([5.0])
    combined = basis(r) * env(r).unsqueeze(-1)
    assert torch.allclose(combined, torch.zeros_like(combined), atol=1e-6)
