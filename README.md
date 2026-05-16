# tinymlip

A small, readable, educational implementation of Machine Learning Interatomic Potentials (MLIPs) — the kind of model that powers tools like CHGNet, NequIP, and MACE.

This repo is for learning. The code prioritizes clarity over speed, and every notebook is meant to be read top-to-bottom.

## What you'll learn

By the end of the notebooks, you will have built — from scratch — a working MLIP that predicts the energy and forces of a molecular system. Along the way you'll understand:

1. **Atoms as graphs** — how a 3D molecule or crystal becomes nodes, edges, and edge features
2. **Message passing** — how information flows between atoms through learned interactions
3. **Energy and forces** — why energy is a sum over atoms, and why forces should come from autograd rather than a separate prediction head
4. **Invariant vs. equivariant models** — the conceptual jump from a rotation-invariant model to a rotation-equivariant one, and what equivariance buys you
5. **Periodic systems** — how the same machinery extends from molecules to crystals

The repo ships two models: an **invariant** message-passing model (based on SchNet) and an **equivariant** one (based on PaiNN), trained on the same data so you can compare them directly. Both are simplified reimplementations meant to be read, not drop-in replacements for the originals.

## Notebooks

| # | Notebook | What it covers |
|---|----------|----------------|
| 01 | `atoms_as_graphs.py` | Building a graph from an `Atoms` object; visualizing it |
| 02 | `message_passing.py` | A hand-traced forward pass through one interaction block |
| 03 | `energy_and_forces.py` | Per-atom energies, sum readout, autograd forces |
| 04 | `training_invariant.py` | Training the invariant model (SchNet-based) on rMD17 |
| 05 | `equivariant_model.py` | Same dataset, equivariant model (PaiNN-based), side-by-side comparison |
| 06 | `crystals_and_pbc.py` | Periodic boundary conditions, using the model as an ASE calculator |

Each notebook runs end-to-end in under 5 minutes on CPU at the default `tiny` config.

## Tech stack

- **Python 3.11**, managed with **[uv](https://docs.astral.sh/uv/)**
- **PyTorch** + **PyTorch Geometric** for models and graph utilities
- **ASE** and **pymatgen** for atomistic I/O
- **polars** for dataset metadata and run logs
- **[marimo](https://marimo.io/)** for notebooks — reactive, stored as plain `.py` files, version-controllable
- **plotly** for interactive plots, **py3Dmol** for structure views

## Getting started

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) installed. If you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone https://github.com/<you>/tinymlip.git
cd tinymlip
uv sync
```

`uv sync` reads `.python-version` (pinned to 3.11), provisions a matching interpreter if needed, creates a virtual environment, and installs the package in editable mode along with all dependencies.

Pull the small starter dataset:

```bash
uv run python data/download.py --dataset rmd17 --molecule aspirin --n-frames 1000
```

Open the first notebook:

```bash
uv run marimo edit notebooks/01_atoms_as_graphs.py
```

## Project layout

```
tinymlip/
├── src/tinymlip/         # the package — graph, layers, models, training
├── notebooks/            # marimo notebooks, in teaching order
├── tests/                # invariance, extensivity, autograd-force tests
├── data/                 # tiny examples + download script for larger datasets
├── configs/              # tiny / small / default training presets
├── pyproject.toml
└── .python-version       # 3.11
```

The package uses the **`src/` layout**, which means `tinymlip` must be installed (which `uv sync` does for you) — you can't just `import tinymlip` from the repo root. This is intentional: it catches packaging issues early and matches modern Python practice.

## Common commands

```bash
# Environment
uv sync                              # install / refresh dependencies
uv add <package>                     # add a runtime dependency
uv add --dev <package>               # add a dev dependency

# Notebooks
uv run marimo edit notebooks/<file>  # edit a notebook
uv run marimo run notebooks/<file>   # run a notebook as a read-only app

# Testing and linting
uv run pytest                        # run the test suite
uv run ruff check .                  # lint
uv run ruff format .                 # format
```

## What this repo is not

- **Not a production MLIP library.** For real work, use [MACE](https://github.com/ACEsuit/mace), [NequIP](https://github.com/mir-group/nequip), [CHGNet](https://github.com/CederGroupHub/chgnet), or [SevenNet](https://github.com/MDIL-SNU/SevenNet). These are faster, more accurate, and battle-tested.
- **Not a survey.** It teaches one path — invariant message passing, then equivariant — clearly, rather than reviewing the whole literature.
- **Not pretrained at scale.** Models are trained per-molecule on small datasets so things run on a laptop. Universal MLIPs require much more data and compute.

## References and further reading

- Schütt et al., *SchNet — A deep learning architecture for molecules and materials* (2018)
- Schütt et al., *Equivariant message passing for the prediction of tensorial properties and molecular spectra* — PaiNN (2021)
- Batzner et al., *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials* — NequIP (2022)
- Batatia et al., *MACE: Higher order equivariant message passing neural networks* (2022)
- Deng et al., *CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling* (2023)
- Christensen & von Lilienfeld, *On the role of gradients for machine learning of molecular energies and forces* — the rMD17 dataset (2020)

## License

MIT.
