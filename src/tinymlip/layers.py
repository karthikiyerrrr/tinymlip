"""Message-passing interaction layers.

Two sibling classes (no shared base — the comparison between invariant and
equivariant message passing IS the lesson):

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


class EquivariantInteraction(nn.Module):
    """One PaiNN-style equivariant message-passing step.

    Based on Schütt et al. 2021 (PaiNN); MPNN framework from Gilmer et al. 2017.

    Each atom carries two feature tensors:
      s : [N, F]        scalar features (rotation-invariant)
      v : [N, F, 3]     vector features (rotate with the molecule)

    Two phases per step:
      1. Message phase: gather scalar messages + two kinds of vector messages
         (propagation of existing v along edges, creation of new v from edge
         directions).
      2. Update phase: per-atom mixing of s and v through the rotation-
         invariants ||V·v|| and <U·v, V·v>, with a gated update on v.

    Deviations from PaiNN:
      - Norm computed as Vv.norm(dim=-1) (no epsilon guard). The reference
        adds a small epsilon inside the sqrt for numerical stability in
        production; we omit it here so the formula reads cleanly. For the
        tiny test molecules used in the notebooks this is never an issue.
      - The update-phase vector mixers are two separate nn.Linear modules
        (U and V), where upstream uses a single Linear(F, 2F, bias=False)
        whose output is split into mu_W (≙ our U·v) and mu_V (≙ our V·v).
        Mathematically identical (same parameter count, same expressivity,
        no bias either way); the split form makes U and V nameable in
        notebook prose. Initialization variance may differ by a small
        constant factor since each nn.Linear seeds its weights independently.

    Implementers: cross-check both phases against the reference PaiNN code in
    https://github.com/atomistic-machine-learning/schnetpack
    (src/schnetpack/representation/painn.py). The reference splits the
    message and update phases into separate nn.Modules (`PaiNNInteraction`
    and `PaiNNMixing`); we fuse them into one forward() for readability.
    Op ordering and shapes inside each phase match the reference; deviations
    are listed above.

    Initialization convention: v is conventionally initialized as zeros at
    the network input (the layer itself does not initialize v; whoever
    calls the layer the first time provides v=0). On the first layer call
    this makes the propagation message a no-op — vectors only enter via
    the creation message until the second layer. This matches PaiNN.
    """

    def __init__(self, hidden_dim: int, num_basis: int, cutoff: float) -> None:
        super().__init__()
        self.basis = BesselBasis(num_basis, cutoff)
        self.envelope = CosineEnvelope(cutoff)

        # Radial filter: maps RBF -> 3 channels (split equally), no nonlinearity
        # (matches PaiNN). Channels: scalar message, vector propagation, vector
        # creation from edge direction.
        self.filter_net = nn.Linear(num_basis, 3 * hidden_dim)

        # Per-edge scalar MLP on sender's s, also split 3 ways.
        self.psi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

        # Update-phase vector mixers. NO BIAS: a constant vector would break
        # rotation equivariance (it wouldn't rotate with the molecule).
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)  # noqa: N806 — U, V match PaiNN paper notation
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=False)  # noqa: N806

        # Update-phase MLP on [s, ||V·v||] -> 3 channels (split equally):
        #   a_ss: scalar correction from vector norms
        #   a_sv: scalar correction from <U·v, V·v>
        #   a_vv: gate (scalar per channel) multiplying U·v
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        graph: AtomGraph,
    ) -> tuple[Tensor, Tensor]:
        # s: [N, hidden_dim]
        # v: [N, hidden_dim, 3]
        # returns (s', v') with the same shapes.

        # ---- Message phase ----
        src, dst = graph.edge_index
        edge_vec = graph.pos[dst] - graph.pos[src]  # autograd: pos is leaf; [E, 3]
        r = edge_vec.norm(dim=-1).clamp(min=1e-6)  # [E]
        unit = edge_vec / r.unsqueeze(-1)  # [E, 3]

        rbf = self.basis(r) * self.envelope(r).unsqueeze(-1)  # [E, num_basis]
        phi_s, phi_vv, phi_vs = self.filter_net(rbf).chunk(3, dim=-1)  # each [E, F]
        psi_s, psi_vv, psi_vs = self.psi(s)[src].chunk(3, dim=-1)  # each [E, F]

        m_s = psi_s * phi_s  # [E, F]
        # Vector propagation: transport sender's existing v along edges.
        m_vv = (psi_vv * phi_vv).unsqueeze(-1) * v[src]  # [E, F, 3]
        # Vector creation: build new vector from edge direction, weighted by s.
        m_vs = (psi_vs * phi_vs).unsqueeze(-1) * unit.unsqueeze(-2)  # [E, F, 3]

        ds = torch.zeros_like(s).index_add_(0, dst, m_s)  # [N, F]
        dv = torch.zeros_like(v).index_add_(0, dst, m_vv + m_vs)  # [N, F, 3]
        s = s + ds
        v = v + dv

        # ---- Update phase ----
        # Apply U, V as linear maps over the F (channel) dimension. v has shape
        # [N, F, 3]; transpose to [N, 3, F] so nn.Linear hits the F axis, then
        # transpose back.
        Uv = self.U(v.transpose(-1, -2)).transpose(-1, -2)  # noqa: N806 — Uv matches PaiNN paper notation; [N, F, 3]
        Vv = self.V(v.transpose(-1, -2)).transpose(-1, -2)  # noqa: N806 — [N, F, 3]

        # Rotation invariants built from v — these are how scalars learn
        # from vector channels.
        vnorm = Vv.norm(dim=-1)  # [N, F]   ||V·v|| per channel
        vdot = (Uv * Vv).sum(dim=-1)  # [N, F]   <U·v, V·v> per channel

        a_ss, a_sv, a_vv = self.update_mlp(torch.cat([s, vnorm], dim=-1)).chunk(
            3, dim=-1
        )  # each [N, F]

        s = s + a_ss + a_sv * vdot  # scalar gets two corrections
        v = v + a_vv.unsqueeze(-1) * Uv  # gated vector update
        return s, v


class AtomicReadout(nn.Module):
    """Per-atom scalar head. Two-layer MLP F -> F/2 -> 1 with SiLU activation.

    Used by both InvariantMPNN and EquivariantMPNN. Returns [N, 1]; the model
    is responsible for summing over atoms. Keeping the sum outside this module
    makes 'energy = sum_i E_i' visible at the model level, which is the central
    teaching point of notebook 03 (and the reason MLIP energies are size-extensive).

    Based on the Atomwise readout block in SchNetPack
    (src/schnetpack/atomistic/atomwise.py), with n_layers=2 and the default
    pyramidal hidden sizing (F -> F/2 -> 1).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [N, hidden_dim] -> [N, 1]
        return self.mlp(x)
