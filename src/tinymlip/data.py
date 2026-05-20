"""Loaders for atomistic datasets used by tinymlip notebooks.

`load_rmd17` returns a polars metadata table paired with a list of ASE Atoms
objects (one per frame). The two views are deliberately separated: polars
handles the tabular metadata (energies, splits, indexing), and atomistic
information (positions, numbers, forces) stays in ASE Atoms.

`to_torch_dataset` converts an `RMD17Bundle` to a `torch.utils.data.Dataset`
for the training notebooks. The two functions are siblings; the adapter does
not re-read disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ase
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RMD17Bundle:
    """A loaded rMD17 molecule: tabular metadata + parallel ASE structures.

    `meta` has one row per frame; `structures[i]` corresponds to `meta.row(i)`.
    """

    meta: pl.DataFrame
    structures: list[ase.Atoms]


def _default_data_root() -> Path:
    # src/tinymlip/data.py → src/tinymlip → src → <repo root>
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "rmd17"


def _load_split_indices(data_root: Path, split: str, cv_fold: int) -> np.ndarray:
    """Read an rMD17 official CV split CSV from disk."""
    assert 1 <= cv_fold <= 5, f"cv_fold must be in 1..5, got {cv_fold}"
    csv_path = data_root / "splits" / f"index_{split}_{cv_fold:02d}.csv"
    return np.loadtxt(csv_path, dtype=np.int64, ndmin=1)


def load_rmd17(
    molecule: str,
    *,
    split: Literal["train", "test", "all"] = "all",
    cv_fold: int = 1,
    n_frames: int | None = None,
    seed: int = 0,
    data_root: Path | None = None,
) -> RMD17Bundle:
    """Load one rMD17 molecule into a polars+ASE bundle.

    Args:
        molecule: rMD17 molecule name (e.g. "ethanol").
        split: "train" or "test" (per the official rMD17 CV fold), or "all" for
            the union of the two.
        cv_fold: Which of the 5 official splits to use (1..5).
        n_frames: If set, subsample this many frames from the chosen split
            with `numpy.random.default_rng(seed)`. Deterministic given seed.
        seed: RNG seed for `n_frames` subsetting.
        data_root: Directory containing `rmd17_<molecule>.npz` and `splits/`.
            Defaults to `<repo>/data/raw/rmd17`.

    Returns:
        An `RMD17Bundle`. Energies are kcal/mol, positions are Å, forces are
        kcal/mol/Å — all units pass through as shipped by rMD17.
    """
    if data_root is None:
        data_root = _default_data_root()

    npz_path = data_root / f"rmd17_{molecule}.npz"
    if not npz_path.exists():
        cmd = f"uv run python data/download.py --dataset rmd17 --molecule {molecule}"
        raise FileNotFoundError(f"rmd17 {molecule} not found at {npz_path}. Run: {cmd}")

    raw = np.load(npz_path)
    nuclear_charges = raw["nuclear_charges"]  # [n_atoms]
    coords = raw["coords"]  # [n_total, n_atoms, 3]
    energies = raw["energies"]  # [n_total]
    forces = raw["forces"]  # [n_total, n_atoms, 3]
    n_atoms = int(coords.shape[1])

    if split == "all":
        train_idx = _load_split_indices(data_root, "train", cv_fold)
        test_idx = _load_split_indices(data_root, "test", cv_fold)
        indices = np.concatenate([train_idx, test_idx])
        split_labels = np.array(["train"] * len(train_idx) + ["test"] * len(test_idx))
    else:
        indices = _load_split_indices(data_root, split, cv_fold)
        split_labels = np.array([split] * len(indices))

    if n_frames is not None and n_frames < len(indices):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(indices), size=n_frames, replace=False)
        indices = indices[chosen]
        split_labels = split_labels[chosen]

    meta = pl.DataFrame(
        {
            "frame_idx": indices.astype(np.int64),
            "molecule": [molecule] * len(indices),
            "n_atoms": np.full(len(indices), n_atoms, dtype=np.int32),
            "energy": energies[indices].astype(np.float64),
            "split": split_labels.tolist(),
            "cv_fold": np.full(len(indices), cv_fold, dtype=np.int32),
        }
    )

    structures: list[ase.Atoms] = []
    for idx in indices:
        atoms = ase.Atoms(numbers=nuclear_charges, positions=coords[idx])
        atoms.info["energy"] = float(energies[idx])
        atoms.arrays["forces"] = forces[idx].copy()
        structures.append(atoms)

    return RMD17Bundle(meta=meta, structures=structures)


class _RMD17TorchDataset(Dataset):
    """Adapter exposing an `RMD17Bundle` as a torch Dataset.

    Each `__getitem__` returns a dict of tensors:
        - z:         LongTensor[n_atoms] — atomic numbers
        - pos:       FloatTensor[n_atoms, 3] — positions
        - energy:    FloatTensor[] — scalar energy
        - forces:    FloatTensor[n_atoms, 3]
        - frame_idx: LongTensor[] — index into the original rMD17 npz

    No batching is done here; notebooks define their own `collate_fn` once
    `graph.py` exists.
    """

    def __init__(self, bundle: RMD17Bundle, dtype: torch.dtype = torch.float32) -> None:
        self._bundle = bundle
        self._dtype = dtype

    def __len__(self) -> int:
        return len(self._bundle.structures)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        atoms = self._bundle.structures[idx]
        frame_idx = int(self._bundle.meta["frame_idx"][idx])
        energy = float(self._bundle.meta["energy"][idx])
        return {
            "z": torch.as_tensor(atoms.numbers, dtype=torch.long),
            "pos": torch.as_tensor(atoms.positions, dtype=self._dtype),
            "energy": torch.tensor(energy, dtype=self._dtype),
            "forces": torch.as_tensor(atoms.arrays["forces"], dtype=self._dtype),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        }


def to_torch_dataset(bundle: RMD17Bundle, *, dtype: torch.dtype = torch.float32) -> Dataset:
    """Wrap an `RMD17Bundle` as a `torch.utils.data.Dataset`.

    The adapter does not re-read disk; it views the already-loaded bundle.
    """
    return _RMD17TorchDataset(bundle, dtype=dtype)
