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

| # | Notebook | What it covers | Online |
|---|----------|----------------|--------|
| 01 | `atoms_as_graphs.py` | Building a graph from an `Atoms` object; live cutoff slider with coupled 3D view and edge-distance histogram | [open](https://molab.marimo.io/github/karthikiyerrrr/tinymlip/blob/main/notebooks/01_atoms_as_graphs.py) |
| 02 | `message_passing.py` | Building a naive MPNN to a SchNet-style `InvariantInteraction` layer; reactive sliders for cutoff and `num_basis`; autograd-derived receptive-field heatmap | [open](https://molab.marimo.io/github/karthikiyerrrr/tinymlip/blob/main/notebooks/02_message_passing.py) |
| 03 | `energy_and_forces.py` | Wrapping `InvariantInteraction` in `InvariantMPNN`; per-atom energy bar chart; force arrows from `torch.autograd.grad`; numerical-gradient check | [open](https://molab.marimo.io/github/karthikiyerrrr/tinymlip/blob/main/notebooks/03_energy_and_forces.py) |
| 04 | `training_invariant.py` | Training `InvariantMPNN` on rMD17 with batched disjoint-union graphs and an energy + force-matching loss; per-element reference shift, learning curves, parity plots, and predicted-vs-true force arrows | [open](https://molab.marimo.io/github/karthikiyerrrr/tinymlip/blob/main/notebooks/04_training_invariant.py) |
| 05 | `equivariant_model.py` | Motivates equivariance with a live rotation demo (forces rotate, scalars stay flat); dissects PaiNN's three message types (scalar, vector propagation, vector creation); trains `EquivariantMPNN` on rMD17 head-to-head against `InvariantMPNN` under identical hyperparameters and compares validation MAE | [open](https://molab.marimo.io/github/karthikiyerrrr/tinymlip/blob/main/notebooks/05_equivariant_model.py) |
| 06 | `crystals_and_pbc.py` | Periodic boundary conditions, using the model as an ASE calculator | [open](https://molab.marimo.io/github/karthikiyerrrr/tinymlip/blob/main/notebooks/06_crystals_and_pbc.py) |

### Reading the notebooks online

GitHub's web preview shows these as flat Python scripts because they're [marimo](https://marimo.io/) `.py` files, not Jupyter `.ipynb`. To see the rendered notebook (cell outputs, plots, structure views) without cloning, click any **Online** link in the table above — they point to [molab](https://molab.marimo.io), marimo's first-party preview service. The URL pattern is `github.com/karthikiyerrrr/tinymlip/blob/main/notebooks/<file>` → `molab.marimo.io/github/karthikiyerrrr/tinymlip/blob/main/notebooks/<file>`. Append `/wasm` to any molab URL for a fully interactive in-browser session (loads Pyodide; slower first paint).

The rendered outputs come from cached session snapshots committed under `notebooks/__marimo__/session/`. When notebook code changes, the snapshots are refreshed locally with `uv run marimo export session notebooks/` and committed alongside the `.py` edits.

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

Pull a starter molecule from rMD17 (ethanol is the smallest, ≈67 MB, one-time download):

```bash
uv run python data/download.py --dataset rmd17 --molecule ethanol
```

The notebooks default to ethanol so they fit the 5-minute-CPU budget. To reproduce literature numbers (SchNet/PaiNN/MACE), download `aspirin` instead — same command, larger molecule.

Frame subsetting happens at load time, not download time — pass `n_frames=` to the loader in a notebook:

```python
from tinymlip.data import load_rmd17
bundle = load_rmd17("ethanol", split="train", cv_fold=1, n_frames=1000)
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
