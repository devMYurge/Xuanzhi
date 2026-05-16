"""Reusable Streamlit render helpers.

Keeping the rendering vocabulary in one module means the views read like
a storyboard and the visual style stays consistent.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from xuanzhi.db import Store
from xuanzhi.schema import Paper

# Colour per edge family for the graph view.
EDGE_COLOURS = {
    "shared_concept": "#4C9AFF",
    "shared_author": "#57D9A3",
    "shared_area": "#FFAB00",
    "citation": "#FF5630",
    "similar_embedding": "#9F7AEA",
    "mixed": "#8993A4",
}

# Colour per source for graph nodes.
SOURCE_COLOURS = {
    "arxiv": "#B3261E",
    "semantic_scholar": "#1A73E8",
    "openalex": "#188038",
    "google_scholar": "#E37400",
    "pubmed": "#7B1FA2",
    "biorxiv": "#00838F",
    "manual": "#5F6368",
}


def paper_card(paper: Paper, *, expanded: bool = False) -> None:
    """Render a compact paper summary block."""
    meta_bits = []
    if paper.year:
        meta_bits.append(str(paper.year))
    meta_bits.append(paper.source.value)
    if paper.citation_count is not None:
        meta_bits.append(f"{paper.citation_count} citations")
    st.markdown(f"**{paper.title}**")
    st.caption(" · ".join(meta_bits))
    if paper.authors:
        names = ", ".join(a.name for a in paper.authors[:6])
        if len(paper.authors) > 6:
            names += f" (+{len(paper.authors) - 6} more)"
        st.write(f":grey[{names}]")
    if paper.research_areas:
        st.write(" ".join(f"`{a.name}`" for a in paper.research_areas))
    if paper.abstract:
        if expanded:
            st.write(paper.abstract)
        else:
            st.caption(
                paper.abstract[:280] + ("…" if len(paper.abstract) > 280 else "")
            )
    if paper.url:
        st.markdown(f"[Open source page]({paper.url})")


def summaries_block(store: Store, paper: Paper) -> None:
    """Show every stored summary for a paper, side by side per model."""
    summaries = store.get_summaries_for_paper(paper.id)
    if not summaries:
        st.info(
            "No summaries yet. Run `python scripts/run_summarise.py` "
            "(add `--claude` to compare against a frontier model)."
        )
        return
    cols = st.columns(len(summaries))
    for col, summary in zip(cols, summaries):
        with col:
            st.markdown(f"**{summary.model}**")
            st.write(summary.summary_text)


def figure_grid(store: Store, paper: Paper, *, columns: int = 3) -> None:
    """Render a paper's extracted figures with type + caption."""
    figures = list(store.iter_figures(paper_id=paper.id))
    if not figures:
        st.info(
            "No figures extracted yet. Run "
            "`python scripts/run_extract_figures.py`."
        )
        return
    rows = [figures[i : i + columns] for i in range(0, len(figures), columns)]
    for row in rows:
        cols = st.columns(columns)
        for col, fig in zip(cols, row):
            with col:
                img_path = Path(fig.image_path)
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.caption("_(image file missing)_")
                st.caption(
                    f"`{fig.figure_type.value}`"
                    + (f" · p.{fig.page_num}" if fig.page_num else "")
                )
                if fig.caption:
                    st.caption(fig.caption[:160])


def render_graph(graph, *, height: int = 600) -> str | None:
    """Render a networkx graph with streamlit-agraph.

    Returns the id of a clicked node (or ``None``). Falls back to a plain
    edge table if streamlit-agraph is not installed, so the view never
    hard-crashes.
    """
    try:
        from streamlit_agraph import Config, Edge, Node, agraph
    except ImportError:
        st.warning(
            "`streamlit-agraph` is not installed — showing an edge table "
            "instead. `pip install streamlit-agraph` for the interactive graph."
        )
        st.dataframe(
            [
                {
                    "source": graph.nodes[a].get("title", a)[:50],
                    "target": graph.nodes[b].get("title", b)[:50],
                    "type": d.get("edge_type"),
                    "weight": d.get("weight"),
                }
                for a, b, d in graph.edges(data=True)
            ],
            use_container_width=True,
        )
        return None

    nodes = []
    for node_id, data in graph.nodes(data=True):
        source = data.get("source", "manual")
        # Size hubs by degree so the layout reads at a glance.
        degree = graph.degree(node_id)
        nodes.append(
            Node(
                id=node_id,
                label=_short(data.get("title", node_id)),
                size=12 + min(degree, 20),
                color=SOURCE_COLOURS.get(source, "#5F6368"),
                title=data.get("title", node_id),
            )
        )
    edges = []
    for a, b, data in graph.edges(data=True):
        etype = data.get("edge_type", "shared_concept")
        edges.append(
            Edge(
                source=a,
                target=b,
                color=EDGE_COLOURS.get(etype, "#8993A4"),
                title=data.get("evidence", etype),
            )
        )
    config = Config(
        height=height,
        width="100%",
        directed=False,
        physics=True,
        nodeHighlightBehavior=True,
        collapsible=False,
    )
    return agraph(nodes=nodes, edges=edges, config=config)


def edge_legend() -> None:
    """A small inline legend for the edge colours."""
    chips = " ".join(
        f"<span style='color:{c}'>&#9632;</span> {name.replace('_', ' ')}"
        for name, c in EDGE_COLOURS.items()
    )
    st.markdown(chips, unsafe_allow_html=True)


def _short(text: str, length: int = 40) -> str:
    text = text or ""
    return text if len(text) <= length else text[: length - 1] + "…"


def empty_state(message: str, *, hint: str | None = None) -> None:
    """Consistent 'no data yet' panel."""
    st.info(message)
    if hint:
        st.code(hint, language="bash")
