"""Shared pytest fixtures for tinymlip tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def rmd17_mini_root() -> Path:
    """Path to the synthetic rMD17 fixture used by data loader tests."""
    return Path(__file__).parent / "fixtures" / "rmd17_mini"
