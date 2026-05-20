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

    **Dataset.** We use rMD17: small organic molecules (here, ethanol)
    with DFT-quality energies and forces. We'll train on these in
    notebook 04; for now we just need a structure to look at.

    **Why a graph, not an MLP on flattened coordinates?** Three reasons
    that recur for the rest of this repo:

    - **Permutation invariance.** Atom ordering is an arbitrary label.
      An MLP on flattened `(x, y, z)` triples is sensitive to it; a graph
      operation that aggregates over neighbours isn't.
    - **Variable system size.** MLPs need a fixed input dimension. rMD17
      molecules vary, crystals vary by orders of magnitude. Graphs scale
      with N for free.
    - **Locality.** Chemistry is local: forces fall off quickly with
      distance. A radial cutoff bakes that physical prior straight into
      the architecture.

    **Prerequisites.** Comfortable with Python and basic `torch` tensors.

    **By the end you can:**
    - Build a graph from an `ase.Atoms` object with `tinymlip.build_graph`.
    - Explain in one sentence what the cutoff parameter does and which
      pairs it keeps.
    - Read `edge_index` and recognize its `[2, E]` shape.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Where does the cutoff come from?** It's a familiar knob from
    classical atomistic modelling — EAM, Tersoff, and DFT real-space
    cutoffs all set one. The relevant length scales:

    - **~1–2 Å**: covalent bonds (C–H ≈ 1.1, C–C ≈ 1.5, C=O ≈ 1.2).
      Below ~1.5 Å the graph sees almost nothing.
    - **~2.5–4 Å**: van der Waals contacts and hydrogen bonds.
      This is what a 5 Å default actually captures.
    - **Above ~6 Å**: edge count grows like O(N²); the model loses its
      locality prior, and training cost balloons.

    The default below is 5 Å. The slider lets you watch the graph respond.
    """)
    return


@app.cell
def _():
    from tinymlip import load_rmd17

    bundle = load_rmd17("ethanol", split="train", n_frames=1, seed=0)
    atoms = bundle.structures[0]
    atoms
    return (atoms,)


@app.cell(hide_code=True)
def _(mo):
    cutoff = mo.ui.slider(start=1.0, stop=6.0, step=0.1, value=5.0, label="cutoff (Å)")
    cutoff
    return (cutoff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every time you move the slider above, the graph is rebuilt. The 3D
    view on the left shows the molecule with its covalent bonds (solid
    grey) and the current graph edges (dashed teal) overlaid. The
    histogram on the right shows the distribution of pairwise distances;
    the orange line is the cutoff. Bars to the left of the line are
    pairs that survive as edges.

    **Graph edges are not chemical bonds.** The cutoff captures every
    pair within a radius, including non-bonded neighbours (hydrogen
    bonds, van der Waals contacts). The graph has no notion of bond
    order, hybridization, or which pairs share electrons — just "are
    these two atoms close enough to interact?" The covalent bonds drawn
    in the 3D view are a chemist's overlay; the edges the model actually
    uses are the dashed teal ones.

    **Where does chemistry enter, then?** Through the atomic numbers `z`
    stored on each node — the same `z` the 3D viewer is using to colour
    atoms by element (carbon dark, oxygen red, hydrogen white). Geometry
    lives in `pos`; chemistry lives in `z`. Both feed the model.
    """)
    return


@app.cell
def _(atoms, cutoff, mo):
    from tinymlip import build_graph, plot_edge_distance_histogram, plot_graph_3d

    graph = build_graph(atoms, cutoff=cutoff.value)

    mo.hstack(
        [
            mo.as_html(plot_graph_3d(graph)),
            mo.as_html(plot_edge_distance_histogram(graph)),
        ],
        widths=[1, 1],
    )
    return (graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Try the slider.** Two cutoff values worth visiting:

    - **Drop to 1.2 Å** — which edges survive, and which physical
      interactions do you lose?

      <details><summary>Expected answer</summary>
      Only the shortest covalent bonds remain (C–H ≈ 1.1 Å); even the
      C–C and C–O backbone (~1.4–1.5 Å) starts dropping out. You've
      lost every non-bonded interaction and most of the covalent
      skeleton. A model trained at this cutoff can't see the molecule
      as a connected object.
      </details>

    - **Push to 6 Å** — what's the edge count, and would this still
      feel local?

      <details><summary>Expected answer</summary>
      Ethanol is about 4 Å across, so the graph is already fully
      connected by the default cutoff of 5 Å: 9 atoms ⇒ 9 × 8 = 72
      directed edges (the maximum). Pushing to 6 Å changes nothing —
      the slider visibly stops doing anything past ~4 Å. The "locality"
      of message passing has already degenerated into "every atom talks
      to every other atom." For ethanol that doesn't matter, but for a
      100-atom protein fragment or a periodic crystal it does — the
      cutoff is the only thing preventing O(N²) explosion and
      overfitting through over-connectivity.
      </details>
    """)
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
                r"""
                **A look at `edge_index`.** Every column is one directed edge
                `(src → dst)`. Because edges are bidirectional, every pair
                appears twice — once in each direction. The reason: in
                notebook 02 the model will aggregate messages *from* each
                atom's neighbours *into* that atom, so each atom needs to be
                a destination for every one of its sources.

                The graph also caches two more arrays alongside `edge_index`:
                `edge_dist` (a scalar distance per edge — what you see in the
                table) and `edge_vec` (the full Cartesian displacement
                `pos[dst] - pos[src]`). Why both? **Distances alone are
                rotation-invariant** — that's the input to the SchNet-style
                model in notebook 04. **Full vectors are needed for the
                equivariant PaiNN-style model in notebook 05**, where the
                network itself learns directional features that rotate with
                the molecule.
                """
            ),
            peek,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What's next.** Notebook 02 puts learnable interactions on these
    edges and runs one forward pass through a message-passing block, so
    you can see how information flows between neighbouring atoms.

    For the materials scientists: `build_graph` currently raises
    `NotImplementedError` for periodic systems. Periodic boundary
    conditions and crystal graphs arrive in notebook 06.
    """)
    return


if __name__ == "__main__":
    app.run()
