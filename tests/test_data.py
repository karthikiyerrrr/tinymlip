"""Tests for tinymlip.data — the rMD17 loader and torch adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tinymlip.data import load_rmd17, to_torch_dataset


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


def test_load_rmd17_energy_and_forces_round_trip(rmd17_mini_root: Path) -> None:
    bundle = load_rmd17("aspirin", split="all", cv_fold=1, data_root=rmd17_mini_root)
    raw = np.load(rmd17_mini_root / "rmd17_aspirin.npz")

    for row_i, atoms in enumerate(bundle.structures):
        frame_idx = bundle.meta["frame_idx"][row_i]
        assert atoms.info["energy"] == pytest.approx(float(raw["energies"][frame_idx]))
        np.testing.assert_allclose(atoms.arrays["forces"], raw["forces"][frame_idx])
        assert atoms.arrays["forces"].shape == (len(atoms), 3)
        assert bundle.meta["energy"][row_i] == pytest.approx(
            float(raw["energies"][frame_idx])
        )


def test_load_rmd17_missing_file_raises_with_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        load_rmd17("aspirin", data_root=tmp_path)

    msg = str(excinfo.value)
    assert "data/download.py --dataset rmd17 --molecule aspirin" in msg
    assert "rmd17_aspirin.npz" in msg


def test_to_torch_dataset_shapes_and_dtypes(rmd17_mini_root: Path) -> None:
    bundle = load_rmd17("aspirin", split="all", cv_fold=1, data_root=rmd17_mini_root)
    ds = to_torch_dataset(bundle)

    assert len(ds) == len(bundle.meta)

    sample = ds[0]
    n_atoms = int(bundle.meta["n_atoms"][0])

    assert sample["z"].dtype == torch.long
    assert sample["z"].shape == (n_atoms,)

    assert sample["pos"].dtype == torch.float32
    assert sample["pos"].shape == (n_atoms, 3)

    assert sample["energy"].dtype == torch.float32
    assert sample["energy"].shape == ()

    assert sample["forces"].dtype == torch.float32
    assert sample["forces"].shape == (n_atoms, 3)

    assert sample["frame_idx"].dtype == torch.long
    assert sample["frame_idx"].shape == ()
    assert int(sample["frame_idx"]) == int(bundle.meta["frame_idx"][0])
