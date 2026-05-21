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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ase
import ase.build
import ase.io
import numpy as np
import polars as pl
import torch
from ase.calculators.emt import EMT
from ase.stress import voigt_6_to_full_3x3_stress
from torch.utils.data import Dataset

from tinymlip.graph import build_graph, collate_graphs


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


def make_collate(
    cutoff: float,
    dtype: torch.dtype = torch.float32,
) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    """Build a `collate_fn` for `torch.utils.data.DataLoader`.

    The dataset yields per-frame dicts (see `_RMD17TorchDataset.__getitem__`);
    `make_collate(cutoff)` returns a function that turns a list of them into a
    single batched dict ready for the training loop:

        {
            "graph":   AtomGraph (batched, .batch populated),
            "energy":  [B] float,
            "forces":  [N_total, 3] float,   # concatenated across frames
            "n_atoms": [B] long,              # for per-atom normalization in the loss
        }

    Graphs are built here (not in the dataset's __getitem__) so cutoff stays a
    DataLoader-time parameter — sliders in the notebook can change it without
    re-reading the dataset.
    """

    def _collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        graphs = []
        for sample in samples:
            atoms_kwargs = dict(
                numbers=sample["z"].cpu().numpy(),
                positions=sample["pos"].cpu().numpy(),
            )
            if "cell" in sample:
                atoms_kwargs["cell"] = sample["cell"].cpu().numpy()
                atoms_kwargs["pbc"] = (
                    [bool(b) for b in sample["pbc"].cpu().numpy()] if "pbc" in sample else True
                )
            atoms = ase.Atoms(**atoms_kwargs)
            graphs.append(build_graph(atoms, cutoff=cutoff, dtype=dtype))
        batched_graph = collate_graphs(graphs)
        out = {
            "graph": batched_graph,
            "energy": torch.stack([s["energy"] for s in samples]).to(dtype),
            "forces": torch.cat([s["forces"] for s in samples], dim=0).to(dtype),
            "n_atoms": torch.tensor([int(s["z"].numel()) for s in samples], dtype=torch.long),
        }
        if "stress" in samples[0]:
            out["stress"] = torch.stack([s["stress"] for s in samples], dim=0).to(dtype)
        return out

    return _collate


def _apply_strain(atoms: ase.Atoms, eps: np.ndarray) -> ase.Atoms:
    """Apply a symmetric strain ε to both positions and cell: r → r·(I+ε), c → c·(I+ε).

    Returns a new Atoms object; the input is not modified.
    """
    i_plus_eps = np.eye(3) + eps
    new = atoms.copy()
    new.set_cell(atoms.cell.array @ i_plus_eps, scale_atoms=False)
    new.set_positions(atoms.positions @ i_plus_eps)
    return new


def generate_cu_dataset(
    *,
    n_snapshots: int,
    supercell: tuple[int, int, int] = (2, 2, 2),
    rattle_amp: float = 0.1,
    strain_range: float = 0.05,
    shear_range: float = 0.02,
    seed: int = 0,
    a: float = 3.615,
) -> list[ase.Atoms]:
    """Generate rattled + strained FCC-Cu snapshots labeled by ASE's EMT.

    Each snapshot starts from FCC Cu at lattice constant `a` × `supercell`,
    then applies in order:
      1. Rattle: per-atom Gaussian displacements with std=`rattle_amp` Å.
      2. Symmetric strain ε with diagonal ~ U(−strain_range, +strain_range)
         and off-diagonal ~ U(−shear_range, +shear_range), applied to both
         positions and cell so the deformation is volume- and shape-consistent.
      3. Attach EMT(), read energy / forces / stress, stash on .info / .arrays.

    Args:
        n_snapshots: number of snapshots to generate.
        supercell:   integer scaling of the conventional FCC cell. (2,2,2) → 32 atoms.
        rattle_amp:  Gaussian rattle std in Å.
        strain_range: half-width of the uniform distribution on diagonal ε.
        shear_range:  half-width on off-diagonal ε.
        seed:        seeds numpy's RNG used for rattle and ε.
        a:           lattice constant in Å. Default 3.615 (EMT Cu equilibrium).

    Returns:
        list of ASE Atoms with .info["energy"] (float), .info["stress"] ([3,3]
        ndarray), .arrays["forces"] ([N,3] ndarray) attached.
    """
    rng = np.random.default_rng(seed)
    base = ase.build.bulk("Cu", "fcc", a=a, cubic=True).repeat(supercell)
    snapshots: list[ase.Atoms] = []
    for _ in range(n_snapshots):
        atoms = base.copy()
        # 1. Rattle (uses its own RNG seed for reproducibility — derive from `rng`)
        atoms.rattle(stdev=rattle_amp, seed=int(rng.integers(0, 2**31 - 1)))
        # 2. Build symmetric strain and apply
        eps = np.zeros((3, 3))
        eps[0, 0] = rng.uniform(-strain_range, strain_range)
        eps[1, 1] = rng.uniform(-strain_range, strain_range)
        eps[2, 2] = rng.uniform(-strain_range, strain_range)
        s01 = rng.uniform(-shear_range, shear_range)
        s02 = rng.uniform(-shear_range, shear_range)
        s12 = rng.uniform(-shear_range, shear_range)
        eps[0, 1] = eps[1, 0] = s01
        eps[0, 2] = eps[2, 0] = s02
        eps[1, 2] = eps[2, 1] = s12
        atoms = _apply_strain(atoms, eps)
        # 3. Label with EMT
        atoms.calc = EMT()
        e = float(atoms.get_potential_energy())
        f = atoms.get_forces()
        s = atoms.get_stress(voigt=False)  # [3, 3]
        # Detach the calculator so the Atoms object is picklable and serializable
        atoms.calc = None
        atoms.info["energy"] = e
        atoms.info["stress"] = np.asarray(s, dtype=np.float64)  # [3, 3]
        atoms.arrays["forces"] = np.asarray(f, dtype=np.float64)
        snapshots.append(atoms)
    return snapshots


def load_cu_emt(
    *,
    cache_dir: str = "data/cu_emt",
    n_snapshots: int = 800,
    supercell: tuple[int, int, int] = (2, 2, 2),
    rattle_amp: float = 0.1,
    strain_range: float = 0.05,
    shear_range: float = 0.02,
    seed: int = 0,
    a: float = 3.615,
    split_fractions: tuple[float, float, float] = (0.75, 0.125, 0.125),
) -> tuple[pl.DataFrame, list[ase.Atoms]]:
    """Load (or generate-then-cache) the synthetic Cu/EMT dataset.

    On first call: runs `generate_cu_dataset` and writes the snapshots to
    `<cache_dir>/snapshots.extxyz` (ASE extended XYZ — carries cells, per-atom
    forces, and stress natively). Subsequent calls just read the file.

    Returns:
        meta: polars DataFrame with columns id, n_atoms, volume, energy,
              force_norm_max, stress_norm, split ("train"/"val"/"test").
        atoms: parallel list of ASE Atoms with labels on .info/.arrays.
    """
    cache_path = Path(cache_dir) / "snapshots.extxyz"
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        snapshots = generate_cu_dataset(
            n_snapshots=n_snapshots,
            supercell=supercell,
            rattle_amp=rattle_amp,
            strain_range=strain_range,
            shear_range=shear_range,
            seed=seed,
            a=a,
        )
        # ase.io.write with extxyz captures cell, pbc, per-atom forces, info dict
        ase.io.write(str(cache_path), snapshots, format="extxyz")
    atoms_list: list[ase.Atoms] = ase.io.read(str(cache_path), index=":")

    # ASE's extxyz reader maps energy/forces/stress into a SinglePointCalculator
    # rather than back to .info/.arrays.  Re-populate the canonical label fields
    # so the caller sees the same interface as generate_cu_dataset produces.
    for atoms in atoms_list:
        if atoms.calc is not None:
            results = atoms.calc.results
            atoms.info["energy"] = float(results["energy"])
            # Stress comes back from extxyz as a Voigt 6-vector; reshape to 3×3.
            s_voigt = np.asarray(results["stress"])
            atoms.info["stress"] = voigt_6_to_full_3x3_stress(s_voigt)
            atoms.arrays["forces"] = np.asarray(results["forces"], dtype=np.float64)
            atoms.calc = None

    # Build deterministic train/val/test split (no shuffle — generation is already
    # random; identical seeds give identical files).
    n = len(atoms_list)
    n_train = int(split_fractions[0] * n)
    n_val = int(split_fractions[1] * n)
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)

    rows = []
    for i, atoms in enumerate(atoms_list):
        f = atoms.arrays["forces"]
        s = atoms.info["stress"]
        rows.append(
            {
                "id": i,
                "n_atoms": len(atoms),
                "volume": float(atoms.get_volume()),
                "energy": float(atoms.info["energy"]),
                "force_norm_max": float(np.linalg.norm(f, axis=1).max()),
                "stress_norm": float(np.linalg.norm(s)),
                "split": splits[i],
            }
        )
    return pl.DataFrame(rows), atoms_list


class _CuEMTTorchDataset(Dataset):
    """torch.utils.data.Dataset over a list of EMT-labeled Cu Atoms.

    Each __getitem__ returns a dict with PBC-aware keys:
      z [N], pos [N,3], cell [3,3], pbc [3] bool, energy scalar,
      forces [N,3], stress [3,3].
    Designed to feed `make_collate(cutoff)`, which builds the
    PBC graph at DataLoader time.
    """

    def __init__(self, atoms_list: list[ase.Atoms], dtype: torch.dtype = torch.float32) -> None:
        self.atoms_list = atoms_list
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.atoms_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        atoms = self.atoms_list[idx]
        return {
            "z": torch.as_tensor(atoms.numbers, dtype=torch.long),
            "pos": torch.as_tensor(atoms.positions, dtype=self.dtype),
            "cell": torch.as_tensor(atoms.cell.array, dtype=self.dtype),
            "pbc": torch.as_tensor(atoms.pbc, dtype=torch.bool),
            "energy": torch.as_tensor(atoms.info["energy"], dtype=self.dtype),
            "forces": torch.as_tensor(atoms.arrays["forces"], dtype=self.dtype),
            "stress": torch.as_tensor(atoms.info["stress"], dtype=self.dtype),
        }


def to_torch_dataset_cu_emt(
    atoms_list: list[ase.Atoms],
    *,
    dtype: torch.dtype = torch.float32,
) -> Dataset:
    """Wrap a list of EMT-labeled Cu Atoms as a torch Dataset for nb06."""
    return _CuEMTTorchDataset(atoms_list, dtype=dtype)
