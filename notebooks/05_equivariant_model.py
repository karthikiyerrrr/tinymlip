import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 05 — Equivariant Model (PaiNN)

        **What this notebook teaches.** Why pure-scalar (invariant) hidden
        representations leave directional information on the table, and how a
        PaiNN-style equivariant interaction adds vector features per atom — `v`
        — that rotate with the molecule. You'll see live rotation equivariance,
        dissect the message phase, then train the equivariant model on rMD17 and
        compare it head-to-head against the invariant model from nb04.

        **Prerequisites.** You've worked through:
        - **nb02** (message passing on graphs, `index_add_`, `InvariantInteraction` walkthrough), and
        - **nb04** (training loop, energy + force loss, rMD17 loading).

        **By the end you can:** explain what `s` and `v` features are for, read
        PaiNN's message + update phases, verify equivariance live, and compare
        invariant vs equivariant training curves on the same dataset under
        identical hyperparameters.

        **Reference.** Schütt, Unke & Gastegger, *Equivariant message passing
        for the prediction of tensorial properties and molecular spectra*, ICML
        2021 (PaiNN). Deviations from the reference are documented in the
        `EquivariantInteraction` docstring.

        Forward link: **nb06** takes this to crystals with periodic boundary conditions.
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import torch
    from torch.utils.data import DataLoader

    from tinymlip.data import load_rmd17, make_collate, to_torch_dataset
    from tinymlip.forces import compute_forces
    from tinymlip.graph import AtomGraph, build_graph
    from tinymlip.layers import EquivariantInteraction
    from tinymlip.models import EquivariantMPNN, InvariantMPNN
    from tinymlip.train import (
        apply_atomic_reference,
        fit_atomic_reference,
        train,
    )
    from tinymlip.viz import element_color, plot_graph_3d

    return (
        AtomGraph,
        DataLoader,
        EquivariantInteraction,
        EquivariantMPNN,
        InvariantMPNN,
        apply_atomic_reference,
        build_graph,
        compute_forces,
        element_color,
        fit_atomic_reference,
        go,
        load_rmd17,
        make_collate,
        np,
        pl,
        plot_graph_3d,
        to_torch_dataset,
        torch,
        train,
    )


if __name__ == "__main__":
    app.run()
