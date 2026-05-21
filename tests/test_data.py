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
    a = load_rmd17("aspirin", split="all", cv_fold=1, n_frames=3, seed=0, data_root=rmd17_mini_root)
    b = load_rmd17("aspirin", split="all", cv_fold=1, n_frames=3, seed=0, data_root=rmd17_mini_root)
    assert a.meta["frame_idx"].to_list() == b.meta["frame_idx"].to_list()


def test_load_rmd17_n_frames_differs_across_seeds(rmd17_mini_root: Path) -> None:
    a = load_rmd17("aspirin", split="all", cv_fold=1, n_frames=3, seed=0, data_root=rmd17_mini_root)
    b = load_rmd17("aspirin", split="all", cv_fold=1, n_frames=3, seed=1, data_root=rmd17_mini_root)
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
        assert bundle.meta["energy"][row_i] == pytest.approx(float(raw["energies"][frame_idx]))


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


def test_make_collate_returns_batched_dict(ethanol_atoms):
    """make_collate stitches per-frame dicts into a single batched payload."""
    import torch

    from tinymlip.data import make_collate
    from tinymlip.graph import AtomGraph

    # Hand-build 3 per-frame dicts that look like _RMD17TorchDataset.__getitem__.
    samples = []
    for shift in (0.0, 0.01, -0.01):
        pos = torch.as_tensor(ethanol_atoms.get_positions() + shift, dtype=torch.float32)
        samples.append(
            {
                "z": torch.as_tensor(ethanol_atoms.numbers, dtype=torch.long),
                "pos": pos,
                "energy": torch.tensor(-4209.5, dtype=torch.float32),
                "forces": torch.zeros(ethanol_atoms.get_global_number_of_atoms(), 3),
                "frame_idx": torch.tensor(0, dtype=torch.long),
            }
        )

    collate = make_collate(cutoff=5.0)
    batch = collate(samples)

    assert isinstance(batch["graph"], AtomGraph)
    assert batch["graph"].batch is not None
    assert batch["energy"].shape == (3,)
    assert batch["forces"].shape == (3 * 9, 3)
    assert torch.equal(batch["n_atoms"], torch.tensor([9, 9, 9], dtype=torch.long))


def test_generate_cu_dataset_produces_atoms_with_labels(tmp_path):
    """A tiny 4-snapshot run produces ASE Atoms with energy/forces/stress."""
    import numpy as np

    from tinymlip.data import generate_cu_dataset

    snapshots = generate_cu_dataset(
        n_snapshots=4,
        supercell=(1, 1, 1),
        rattle_amp=0.05,
        strain_range=0.02,
        shear_range=0.01,
        seed=0,
    )

    assert len(snapshots) == 4
    for atoms in snapshots:
        assert atoms.cell.array.shape == (3, 3)
        assert all(atoms.pbc)
        assert "energy" in atoms.info
        assert isinstance(atoms.info["energy"], float)
        assert "stress" in atoms.info
        assert atoms.info["stress"].shape == (3, 3)
        assert atoms.arrays["forces"].shape == (len(atoms), 3)
        assert np.isfinite(atoms.info["energy"])
        assert np.isfinite(atoms.info["stress"]).all()
        assert np.isfinite(atoms.arrays["forces"]).all()


def test_load_cu_emt_roundtrips_cached_snapshots(tmp_path):
    """First call generates+caches; second call reads from cache and matches."""
    import numpy as np

    from tinymlip.data import load_cu_emt

    meta1, atoms1 = load_cu_emt(
        cache_dir=str(tmp_path / "cu_emt"),
        n_snapshots=4,
        supercell=(1, 1, 1),
        seed=0,
    )
    meta2, atoms2 = load_cu_emt(
        cache_dir=str(tmp_path / "cu_emt"),
        n_snapshots=4,
        supercell=(1, 1, 1),
        seed=0,
    )
    assert meta1.shape == meta2.shape
    np.testing.assert_allclose(meta1["energy"].to_numpy(), meta2["energy"].to_numpy())
    assert len(atoms1) == len(atoms2)
    np.testing.assert_allclose(atoms1[0].positions, atoms2[0].positions)


def test_make_collate_passes_stress_when_present():
    """If samples carry a 'stress' key, the batched dict has [B, 3, 3] stress."""
    from tinymlip.data import make_collate

    sample_a = {
        "z": torch.tensor([1, 1], dtype=torch.long),
        "pos": torch.zeros(2, 3),
        "energy": torch.tensor(0.0),
        "forces": torch.zeros(2, 3),
        "stress": torch.eye(3, dtype=torch.float32),
        "cell": torch.eye(3, dtype=torch.float32) * 4.0,
        "pbc": torch.tensor([True, True, True]),
    }
    sample_b = {
        "z": torch.tensor([1, 1, 1], dtype=torch.long),
        "pos": torch.zeros(3, 3),
        "energy": torch.tensor(0.0),
        "forces": torch.zeros(3, 3),
        "stress": 2 * torch.eye(3, dtype=torch.float32),
        "cell": torch.eye(3, dtype=torch.float32) * 5.0,
        "pbc": torch.tensor([True, True, True]),
    }
    collate = make_collate(cutoff=2.0)
    batch = collate([sample_a, sample_b])

    assert "stress" in batch
    assert batch["stress"].shape == (2, 3, 3)
    torch.testing.assert_close(batch["stress"][0], torch.eye(3))
    torch.testing.assert_close(batch["stress"][1], 2 * torch.eye(3))
