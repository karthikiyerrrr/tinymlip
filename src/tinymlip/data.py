"""Loaders for atomistic datasets used by tinymlip notebooks.

`load_rmd17` returns a polars metadata table paired with a list of ASE Atoms
objects (one per frame). The two views are deliberately separated: polars
handles the tabular metadata (energies, splits, indexing), and atomistic
information (positions, numbers, forces) stays in ASE Atoms — see CLAUDE.md.

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
    csv_path = data_root / "splits" / f"index_{split}_0{cv_fold}.csv"
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
        molecule: rMD17 molecule name (e.g. "aspirin").
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
        raise FileNotFoundError(
            f"rmd17 {molecule} not found at {npz_path}. Run: {cmd}"
        )

    raw = np.load(npz_path)
    nuclear_charges = raw["nuclear_charges"]  # [n_atoms]
    coords = raw["coords"]                    # [n_total, n_atoms, 3]
    energies = raw["energies"]                # [n_total]
    forces = raw["forces"]                    # [n_total, n_atoms, 3]
    n_atoms = int(coords.shape[1])

    if split == "all":
        train_idx = _load_split_indices(data_root, "train", cv_fold)
        test_idx = _load_split_indices(data_root, "test", cv_fold)
        indices = np.concatenate([train_idx, test_idx])
        split_labels = np.array(
            ["train"] * len(train_idx) + ["test"] * len(test_idx)
        )
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
