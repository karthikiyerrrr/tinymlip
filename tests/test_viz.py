"""Smoke tests for tinymlip.viz."""

from __future__ import annotations


def test_e_v_curve_returns_plotly_figure():
    import plotly.graph_objects as go
    import torch
    from ase import Atoms

    from tinymlip.models import EquivariantMPNN
    from tinymlip.viz import e_v_curve

    a = 3.6
    atoms = Atoms(
        numbers=[29] * 4,
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, a / 2, a / 2],
            [a / 2, 0.0, a / 2],
            [a / 2, a / 2, 0.0],
        ],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True,
    )
    torch.manual_seed(0)
    model = EquivariantMPNN(n_layers=1, hidden_dim=8, num_basis=8, cutoff=4.0).double()

    fig = e_v_curve(model, atoms, volume_fractions=[0.95, 1.0, 1.05])
    assert isinstance(fig, go.Figure)
    # At least one trace (predicted curve); a second trace is added by callers
    # who want the reference comparison.
    assert len(fig.data) >= 1
