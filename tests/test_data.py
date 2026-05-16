"""Tests for tinymlip.data — the rMD17 loader and torch adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinymlip.data import load_rmd17


def test_load_rmd17_returns_paired_meta_and_structures(rmd17_mini_root: Path) -> None:
    bundle = load_rmd17("aspirin", split="all", cv_fold=1, data_root=rmd17_mini_root)

    assert len(bundle.meta) == len(bundle.structures)
    assert len(bundle.meta) == 5  # 3 train + 2 test from the fixture
    for i, atoms in enumerate(bundle.structures):
        assert bundle.meta["n_atoms"][i] == len(atoms)
        assert bundle.meta["molecule"][i] == "aspirin"
