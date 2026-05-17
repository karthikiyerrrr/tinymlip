"""Shared pytest fixtures for tinymlip tests."""

from __future__ import annotations

from pathlib import Path

import ase
import pytest
from ase.build import molecule


@pytest.fixture
def rmd17_mini_root() -> Path:
    """Path to the synthetic rMD17 fixture used by data loader tests."""
    return Path(__file__).parent / "fixtures" / "rmd17_mini"


@pytest.fixture
def ethanol_atoms() -> ase.Atoms:
    """Ethanol (C2H5OH) — 9 atoms. Built from ASE's g2 set so tests do not
    depend on having rMD17 downloaded. Used by force and extensivity tests."""
    return molecule("CH3CH2OH")
