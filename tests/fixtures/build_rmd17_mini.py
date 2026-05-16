"""Build the tiny synthetic rMD17 fixture used by tests/test_data.py.

Run once whenever the fixture shape changes, then commit the generated files:

    uv run python tests/fixtures/build_rmd17_mini.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURE_ROOT = Path(__file__).parent / "rmd17_mini"
N_FRAMES = 5
# A methane-shaped 5-atom molecule (1 C + 4 H). Chemistry is irrelevant to the
# loader; we just need a realistic rMD17-shaped npz.
NUCLEAR_CHARGES = np.array([6, 1, 1, 1, 1], dtype=np.int64)
BASE_POSITIONS = np.array(
    [
        [0.00, 0.00, 0.00],
        [1.09, 0.00, 0.00],
        [-0.36, 1.03, 0.00],
        [-0.36, -0.51, 0.89],
        [-0.36, -0.51, -0.89],
    ],
    dtype=np.float64,
)


def main() -> None:
    rng = np.random.default_rng(seed=42)
    coords = np.stack(
        [BASE_POSITIONS + 0.01 * rng.standard_normal(BASE_POSITIONS.shape) for _ in range(N_FRAMES)]
    )
    energies = (-40.0 + 0.05 * rng.standard_normal(N_FRAMES)).astype(np.float64)
    forces = (0.1 * rng.standard_normal((N_FRAMES, NUCLEAR_CHARGES.size, 3))).astype(np.float64)

    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(
        FIXTURE_ROOT / "rmd17_aspirin.npz",
        nuclear_charges=NUCLEAR_CHARGES,
        coords=coords,
        energies=energies,
        forces=forces,
    )

    splits_dir = FIXTURE_ROOT / "splits"
    splits_dir.mkdir(exist_ok=True)
    (splits_dir / "index_train_01.csv").write_text("0\n1\n2\n")
    (splits_dir / "index_test_01.csv").write_text("3\n4\n")

    print(f"Wrote fixture to {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
