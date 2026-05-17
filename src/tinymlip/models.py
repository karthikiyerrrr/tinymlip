"""Energy models that compose tinymlip's interaction layers.

Two sibling classes (no shared base — the comparison between invariant and
equivariant models IS the lesson, per CLAUDE.md):

  - InvariantMPNN:  SchNet-based. Scalar features per atom.
                    Based on Schütt et al. 2018.
  - EquivariantMPNN: PaiNN-based. Scalar + vector features per atom.
                     Based on Schütt et al. 2021.

Both wrap their interaction layers from tinymlip.layers and a shared
AtomicReadout, and sum the per-atom energies inside `forward`. The sum
lives in the model (not in AtomicReadout) so notebook 03 can show
'readout, then sum' as two distinct operations.

References:
  - Gilmer et al. 2017, "Neural Message Passing for Quantum Chemistry"
  - Schütt et al. 2018, "SchNet — a deep learning architecture for
    molecules and materials", J. Chem. Phys.
  - Schütt et al. 2021, "Equivariant message passing for the prediction
    of tensorial properties and molecular spectra", ICML.

Forces are NOT a method on these models; use `tinymlip.compute_forces`.
This keeps the F = -grad(E) moment visible in caller code and reinforces
that forces are a property of any energy callable, not of a particular
model class.
"""

from __future__ import annotations

from torch import Tensor, nn

from tinymlip.graph import AtomGraph
from tinymlip.layers import (
    AtomicReadout,
    EquivariantInteraction,  # noqa: F401 — re-exported here for Task 5 (EquivariantMPNN)
    InvariantInteraction,
)


class InvariantMPNN(nn.Module):
    """SchNet-based invariant MPNN: embedding -> N InvariantInteractions -> readout -> sum.

    Based on Schütt et al. 2018 (SchNet). Layer-level deviations match those
    documented on InvariantInteraction (Bessel+envelope basis, SiLU, etc.).
    Model-level deviations:
      - No per-element reference energy shift. SchNet subtracts a learned or
        precomputed atomic reference; we omit it here so notebook 03 shows
        the un-augmented sum readout. Training (notebook 04) adds it back.
      - Embedding size tied to hidden_dim. SchNet allows them to differ.
      - Embedding table sized n_elements=100 by default (SchNetPack convention,
        src/schnetpack/representation/schnet.py:121).
      - Our InvariantInteraction.forward returns post-residual features (the
        residual is in the layer, not the model), so this forward loop is
        `x = layer(x, graph)`, not `x = x + layer(...)`. The upstream SchNet
        does the latter because its interaction returns the delta only.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_basis: int,
        cutoff: float,
        n_layers: int,
        n_elements: int = 100,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_elements, hidden_dim)
        self.interactions = nn.ModuleList(
            [InvariantInteraction(hidden_dim, num_basis, cutoff) for _ in range(n_layers)]
        )
        self.readout = AtomicReadout(hidden_dim)

    def forward(self, graph: AtomGraph) -> Tensor:
        # returns scalar energy []
        x = self.embed(graph.z)  # [N, F]
        for layer in self.interactions:
            x = layer(x, graph)  # [N, F]
        per_atom_e = self.readout(x).squeeze(-1)  # [N]
        return per_atom_e.sum()  # []
