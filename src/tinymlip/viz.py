"""Plotly-based visualization helpers for tinymlip graphs.

These helpers take an `AtomGraph` and return a plotly `Figure` (or a markdown
string). They do not import marimo — notebooks compose the helpers with
`mo.ui.*` and layout helpers themselves. This keeps the helpers reusable
across notebooks and easy to smoke-test.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
from ase.data import chemical_symbols, covalent_radii

from tinymlip.graph import AtomGraph

# CPK-ish hex palette for the elements rMD17 actually contains
# (H, C, N, O, F, S, Cl) — anything else falls back to a neutral grey.
_ELEMENT_COLORS: dict[int, str] = {
    1: "#ffffff",  # H
    6: "#444444",  # C
    7: "#3050f8",  # N
    8: "#ff0d0d",  # O
    9: "#90e050",  # F
    16: "#ffff30",  # S
    17: "#1ff01f",  # Cl
}
_DEFAULT_COLOR = "#888888"


def element_color(z: int) -> str:
    """CPK-ish hex color for atomic number `z`. Falls back to neutral grey."""
    return _ELEMENT_COLORS.get(int(z), _DEFAULT_COLOR)


def element_radius(z: int) -> float:
    """Covalent radius of element `z` in Å (from `ase.data.covalent_radii`)."""
    return float(covalent_radii[int(z)])


# Multiplier on the sum of ASE covalent radii used to decide whether a pair
# is "bonded" for visual purposes. 1.15 matches the loose convention used by
# many viewers and tolerates rMD17's slightly stretched optimized geometries.
_BOND_RADIUS_SLOP = 1.15

# Multiplier applied to ASE covalent radii when sizing markers in plotly.
# ASE radii are in Å (~0.3 – 1.0); plotly marker sizes are in pixels.
_MARKER_SIZE_SCALE = 20.0


def _bond_pairs(z: torch.Tensor, pos: torch.Tensor) -> list[tuple[int, int]]:
    """Pairs (i, j) with i < j whose interatomic distance is below a covalent-bond threshold.

    This is for *visual* bonds only — the threshold uses ASE covalent radii
    × ``_BOND_RADIUS_SLOP`` (1.15) and is a common viewer convention. These
    bonds are independent of the graph edges used by the model; the same
    function is called from the same notebook to render both.
    """
    n = z.shape[0]
    radii = np.array([element_radius(int(zi)) for zi in z])
    pos_np = pos.detach().cpu().numpy()
    out: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            threshold = (radii[i] + radii[j]) * _BOND_RADIUS_SLOP
            if float(np.linalg.norm(pos_np[i] - pos_np[j])) <= threshold:
                out.append((i, j))
    return out


def _line_trace(
    pos: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    color: str,
    width: float,
    dash: str | None,
    name: str,
) -> go.Scatter3d:
    """A single Scatter3d trace drawing all (i, j) pairs as broken line segments.

    plotly draws segments as one polyline with NaN gaps separating them.
    """
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for i, j in pairs:
        xs.extend([pos[i, 0], pos[j, 0], np.nan])
        ys.extend([pos[i, 1], pos[j, 1], np.nan])
        zs.extend([pos[i, 2], pos[j, 2], np.nan])
    line: dict[str, object] = {"color": color, "width": width}
    if dash is not None:
        line["dash"] = dash
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=line,
        name=name,
        hoverinfo="skip",
    )


def plot_graph_3d(
    graph: AtomGraph,
    *,
    show_bonds: bool = True,
    show_edges: bool = True,
    height: int = 480,
) -> go.Figure:
    """Render the graph as a plotly 3D scatter.

    Atoms are CPK-ish coloured spheres sized by ASE covalent radius. Bonds
    (when `show_bonds`) are solid grey lines drawn whenever the pair distance
    is below the sum of covalent radii × 1.15. Graph edges (when
    `show_edges`) are dashed teal lines drawn for every directed edge in
    `graph.edge_index` — we draw each undirected pair once.

    Hover on an atom shows its index, element symbol, and degree.
    """
    pos_np = graph.pos.detach().cpu().numpy()
    z_np = graph.z.detach().cpu().numpy()

    # Undirected edge set for drawing — edge_index is bidirectional, so
    # taking i<j collapses each pair to one segment.
    edge_pairs: list[tuple[int, int]] = []
    if show_edges and graph.n_edges > 0:
        seen: set[tuple[int, int]] = set()
        for src, dst in graph.edge_index.t().tolist():
            key = (min(src, dst), max(src, dst))
            if key in seen:
                continue
            seen.add(key)
            edge_pairs.append(key)

    bond_pairs = _bond_pairs(graph.z, graph.pos) if show_bonds else []

    # Per-atom degree for hover text.
    degree = torch.zeros(graph.n_atoms, dtype=torch.long)
    if graph.n_edges > 0:
        degree.scatter_add_(0, graph.edge_index[0], torch.ones(graph.n_edges, dtype=torch.long))
    degree_np = degree.numpy()

    hover_text = [
        f"atom {i} ({chemical_symbols[int(z_np[i])]})<br>degree = {int(degree_np[i])}"
        for i in range(graph.n_atoms)
    ]

    atoms_trace = go.Scatter3d(
        x=pos_np[:, 0],
        y=pos_np[:, 1],
        z=pos_np[:, 2],
        mode="markers",
        marker={
            "size": [element_radius(int(z)) * _MARKER_SIZE_SCALE for z in z_np],
            "color": [element_color(int(z)) for z in z_np],
            "line": {"color": "#222", "width": 1},
        },
        text=hover_text,
        hoverinfo="text",
        name="atoms",
    )

    traces: list[go.Scatter3d] = []
    if bond_pairs:
        traces.append(
            _line_trace(pos_np, bond_pairs, color="#6b7280", width=6, dash=None, name="bonds")
        )
    if edge_pairs:
        traces.append(
            _line_trace(
                pos_np,
                edge_pairs,
                color="#3ddbd9",
                width=2,
                dash="dash",
                name="graph edges",
            )
        )
    traces.append(atoms_trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        scene={
            "xaxis_title": "x (Å)",
            "yaxis_title": "y (Å)",
            "zaxis_title": "z (Å)",
            "aspectmode": "data",
        },
        showlegend=True,
    )
    return fig
