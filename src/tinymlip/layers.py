"""Message-passing interaction layers.

Two sibling classes (no shared base — the comparison between invariant and
equivariant message passing IS the lesson, per CLAUDE.md):

  - InvariantInteraction:  SchNet-style continuous-filter convolution.
                           Based on Schütt et al. 2018.
  - EquivariantInteraction: PaiNN-style block (message phase + update phase).
                            Based on Schütt et al. 2021.

Both compose the shared Bessel + cosine-envelope radial basis from
tinymlip.basis, and both aggregate edge messages via torch.index_add_
directly on the raw AtomGraph — no PyTorch Geometric MessagePassing base
class. The aggregation op stays explicit so notebook 02 can show it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tinymlip.basis import BesselBasis, CosineEnvelope
from tinymlip.graph import AtomGraph


class InvariantInteraction(nn.Module):
    """One SchNet-style invariant message-passing step.

    Based on Schütt et al. 2018 (SchNet); MPNN framework from Gilmer et al. 2017.

    Deviations from SchNet (schnetpack SchNetInteraction):
      - Radial basis is Bessel + cosine envelope (vs SchNet's Gaussian RBF).
        Bessel+envelope is exactly zero at the cutoff, which keeps the
        energy/forces smooth as atoms cross the boundary. SchNet's Gaussians
        rely on the filter network to learn that behavior.
      - SiLU activation (vs SchNet's shifted-softplus). Same monotonic-smooth
        shape, available as torch.nn.SiLU, one fewer custom op to explain.
      - The cosine envelope is multiplied into the radial basis *before* the
        filter network (rbf * envelope → filter_net → W). SchNet multiplies
        the cutoff scalar *after* the filter network (filter_net(f_ij) * rcut_ij).
        Both guarantee W → 0 as r → cutoff; we fuse the envelope into the
        input features so filter_net never sees a discontinuous jump.
      - phi_s (sender projection) uses bias=True (nn.Linear default). SchNet's
        in2f uses bias=False. The bias does not affect invariance properties,
        and the explicit=True default is clearer for a teaching context.
      - filter_net, phi_s, and phi_u all operate in the same hidden_dim
        (equivalent to SchNet with n_filters == n_atom_basis). SchNet allows
        them to differ; we tie them for simplicity.
      - The residual skip connection is computed inside forward() rather than
        in the outer model loop. The net effect is identical.

    Implementers: cross-check forward() against the reference SchNet code in
    https://github.com/atomistic-machine-learning/schnetpack
    (src/schnetpack/representation/schnet.py). If our code disagrees with
    the reference in ways not listed above, that is a bug to fix or a
    deviation to add to this docstring.
    """

    def __init__(self, hidden_dim: int, num_basis: int, cutoff: float) -> None:
        super().__init__()
        self.basis = BesselBasis(num_basis, cutoff)
        self.envelope = CosineEnvelope(cutoff)
        # SchNet-style continuous-filter network: radial features -> per-edge
        # filter of size hidden_dim, which then modulates the scalar message.
        self.filter_net = nn.Sequential(
            nn.Linear(num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # phi_s: pre-filter projection of sender features (no nonlinearity).
        self.phi_s = nn.Linear(hidden_dim, hidden_dim)
        # phi_u: post-aggregation update MLP, applied to the gathered message.
        self.phi_u = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor, graph: AtomGraph) -> Tensor:
        # x: [N, hidden_dim]  -- scalar features per atom
        # returns: [N, hidden_dim]
        src, dst = graph.edge_index  # src=sender j, dst=receiver i

        # Recompute edge_vec from graph.pos so that pos remains the autograd
        # leaf. Notebook 03 will use this for force = -dE/dpos.
        edge_vec = graph.pos[dst] - graph.pos[src]  # [E, 3]
        r = edge_vec.norm(dim=-1).clamp(min=1e-6)  # [E]

        rbf = self.basis(r) * self.envelope(r).unsqueeze(-1)  # [E, num_basis]
        W = self.filter_net(rbf)  # [E, hidden_dim]  # noqa: N806 — W matches SchNet's Wij notation

        # Continuous-filter convolution: project sender features, then
        # element-wise modulate by the radial filter.
        m = self.phi_s(x)[src] * W  # [E, hidden_dim]

        # Hand-rolled scatter-add aggregation at receivers (PyG's scatter does
        # the same; we keep the op explicit so notebook 02 can show it).
        agg = torch.zeros_like(x).index_add_(0, dst, m)  # [N, hidden_dim]
        return x + self.phi_u(agg)  # residual update
