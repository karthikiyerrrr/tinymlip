import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 · Atoms as graphs

    **What this notebook teaches.** An atomic system becomes a graph: every
    atom is a node, every pair within a chosen cutoff radius becomes an edge.
    The cutoff is a knob — moving it changes the graph.

    **Prerequisites.** Comfortable with Python and basic `torch` tensors.

    **By the end you can:**
    - Build a graph from an `ase.Atoms` object with `tinymlip.build_graph`.
    - Explain in one sentence what the cutoff parameter does and which pairs
      it keeps.
    - Read `edge_index` and recognize its `[2, E]` shape.
    """)
    return


@app.cell
def _():
    from tinymlip.data import load_rmd17

    bundle = load_rmd17("ethanol", split="train", n_frames=1, seed=0)
    atoms = bundle.structures[0]
    atoms
    return (atoms,)


@app.cell
def _(mo):
    cutoff = mo.ui.slider(start=1.0, stop=6.0, step=0.1, value=5.0, label="cutoff (Å)")
    cutoff
    return (cutoff,)


@app.cell
def _(atoms, cutoff):
    from tinymlip import build_graph

    graph = build_graph(atoms, cutoff=cutoff.value)
    graph
    return (graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every time you move the slider above, the graph is rebuilt. The 3D view
    on the left shows the molecule with its covalent bonds (solid grey) and
    the current graph edges (dashed teal) overlaid. The histogram on the
    right shows the distribution of pairwise distances; the orange line is
    the cutoff. Bars to the left of the line are pairs that survive as edges.
    """)
    return


@app.cell
def _(graph, mo):
    from tinymlip import plot_edge_distance_histogram, plot_graph_3d

    mo.hstack(
        [
            mo.as_html(plot_graph_3d(graph)),
            mo.as_html(plot_edge_distance_histogram(graph)),
        ],
        widths=[1, 1],
    )
    return


@app.cell
def _(graph, mo):
    from tinymlip import graph_stats_md

    mo.md(graph_stats_md(graph))
    return


@app.cell
def _(graph, mo):
    import polars as pl

    src = graph.edge_index[0].tolist()
    dst = graph.edge_index[1].tolist()
    dist = [round(d, 3) for d in graph.edge_dist.tolist()]
    peek = pl.DataFrame({"src": src, "dst": dst, "dist (Å)": dist}).head(10)

    mo.vstack(
        [
            mo.md(
                "**A look at `edge_index`.** Every column is one directed edge "
                "`(src → dst)`. Because edges are bidirectional, every pair "
                "appears twice — once in each direction."
            ),
            peek,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What's next.** Notebook 02 puts learnable interactions on these edges
    and runs one forward pass through a message-passing block, so you can
    see how information flows between neighbouring atoms.
    """)
    return


if __name__ == "__main__":
    app.run()
