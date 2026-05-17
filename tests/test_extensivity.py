"""Size-extensivity tests for the energy models.

Two copies of ethanol translated by 20 A along x have no edges between them
at any reasonable cutoff (max bond ~1.5 A, vdW radii < 2 A; cutoff 5 A < 20).
The per-atom sum readout therefore must give exactly 2 x E_single (up to
float roundoff). This is a structural property — does not require training.
"""

from __future__ import annotations

import ase
import numpy as np
import torch

from tinymlip.graph import build_graph
from tinymlip.models import EquivariantMPNN, InvariantMPNN


def _duplicate_far(atoms: ase.Atoms, shift: float = 20.0) -> ase.Atoms:
    """Return atoms ++ (atoms + [shift, 0, 0]) as one ASE Atoms object."""
    pos = atoms.get_positions()
    shifted = pos + np.array([shift, 0.0, 0.0])
    new_pos = np.concatenate([pos, shifted], axis=0)
    new_numbers = np.concatenate([atoms.numbers, atoms.numbers], axis=0)
    return ase.Atoms(numbers=new_numbers, positions=new_pos)


def test_invariant_energy_extensive_for_separated_dimer(ethanol_atoms):
    torch.manual_seed(0)
    cutoff = 5.0
    model = InvariantMPNN(hidden_dim=16, num_basis=8, cutoff=cutoff, n_layers=2).double()

    single = build_graph(ethanol_atoms, cutoff=cutoff, dtype=torch.float64)
    e_single = model(single).item()

    dimer_atoms = _duplicate_far(ethanol_atoms, shift=20.0)
    dimer = build_graph(dimer_atoms, cutoff=cutoff, dtype=torch.float64)
    e_dimer = model(dimer).item()

    assert abs(e_dimer - 2 * e_single) < 1e-6, (
        f"E_dimer={e_dimer:.8f}, 2*E_single={2 * e_single:.8f}"
    )


def test_equivariant_energy_extensive_for_separated_dimer(ethanol_atoms):
    torch.manual_seed(0)
    cutoff = 5.0
    model = EquivariantMPNN(hidden_dim=16, num_basis=8, cutoff=cutoff, n_layers=2).double()

    single = build_graph(ethanol_atoms, cutoff=cutoff, dtype=torch.float64)
    e_single = model(single).item()

    dimer_atoms = _duplicate_far(ethanol_atoms, shift=20.0)
    dimer = build_graph(dimer_atoms, cutoff=cutoff, dtype=torch.float64)
    e_dimer = model(dimer).item()

    assert abs(e_dimer - 2 * e_single) < 1e-6, (
        f"E_dimer={e_dimer:.8f}, 2*E_single={2 * e_single:.8f}"
    )
