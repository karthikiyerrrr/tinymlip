"""Plotly-based visualization helpers for tinymlip graphs.

These helpers take an `AtomGraph` and return a plotly `Figure` (or a markdown
string). They do not import marimo — notebooks compose the helpers with
`mo.ui.*` and layout helpers themselves. This keeps the helpers reusable
across notebooks and easy to smoke-test.
"""

from __future__ import annotations

from ase.data import covalent_radii

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
