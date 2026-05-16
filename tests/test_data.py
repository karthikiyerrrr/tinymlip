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


def test_load_rmd17_split_train_selects_correct_indices(rmd17_mini_root: Path) -> None:
    bundle = load_rmd17("aspirin", split="train", cv_fold=1, data_root=rmd17_mini_root)

    assert sorted(bundle.meta["frame_idx"].to_list()) == [0, 1, 2]
    assert set(bundle.meta["split"].to_list()) == {"train"}


def test_load_rmd17_split_test_selects_correct_indices(rmd17_mini_root: Path) -> None:
    bundle = load_rmd17("aspirin", split="test", cv_fold=1, data_root=rmd17_mini_root)

    assert sorted(bundle.meta["frame_idx"].to_list()) == [3, 4]
    assert set(bundle.meta["split"].to_list()) == {"test"}


def test_load_rmd17_n_frames_is_deterministic_under_seed(rmd17_mini_root: Path) -> None:
    a = load_rmd17(
        "aspirin", split="all", cv_fold=1, n_frames=3, seed=0, data_root=rmd17_mini_root
    )
    b = load_rmd17(
        "aspirin", split="all", cv_fold=1, n_frames=3, seed=0, data_root=rmd17_mini_root
    )
    assert a.meta["frame_idx"].to_list() == b.meta["frame_idx"].to_list()


def test_load_rmd17_n_frames_differs_across_seeds(rmd17_mini_root: Path) -> None:
    a = load_rmd17(
        "aspirin", split="all", cv_fold=1, n_frames=3, seed=0, data_root=rmd17_mini_root
    )
    b = load_rmd17(
        "aspirin", split="all", cv_fold=1, n_frames=3, seed=1, data_root=rmd17_mini_root
    )
    assert a.meta["frame_idx"].to_list() != b.meta["frame_idx"].to_list()


def test_load_rmd17_n_frames_caps_at_available(rmd17_mini_root: Path) -> None:
    # Asking for more than available returns all available, no error.
    bundle = load_rmd17(
        "aspirin", split="train", cv_fold=1, n_frames=999, data_root=rmd17_mini_root
    )
    assert len(bundle.meta) == 3
