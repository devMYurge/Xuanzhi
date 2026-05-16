"""Xuanzhi Streamlit app — entry point.

Six views behind a sidebar selector:

    Overview            DB dashboard — counts, sources, areas, concepts.
    Ingest              Add papers live via the Semantic Scholar API.
    Knowledge Graph     networkx graph of the corpus, filterable.
    Paper Explorer      Search papers; per-paper summaries + figures.
    Cross-Literature    Pick two areas → shared concepts + bridging papers.
    Figure Source Lookup  Upload an image → find the source paper to cite.

Every view checks its prerequisites and shows a friendly empty state
with the exact command to run, so the app demos cleanly at any stage.
"""

from __future__ import annotations

import asyncio

import streamlit as st

from xuanzhi.app import components as ui
from xuanzhi.app.data import (
    DEFAULT_DB_PATH,
    db_overview,
    get_store,
    list_papers,
    research_areas,
)

st.set_page_config(page_title="Xuanzhi", page_icon="\U0001F4DA", layout="wide")


# --------------------------------------------------------------- session


def _token() -> int:
    """A cache-busting token; bumped after a live ingest so views refresh."""
    return st.session_state.setdefault("data_token", 0)


def _bump_token() -> None:
    st.session_state["data_token"] = _token() + 1


# ------------------------------------------------------------- sidebar


st.sidebar.title("玄之 Xuanzhi")
st.sidebar.caption("Academic-research cognition layer")

db_path = st.sidebar.text_input("Database path", value=str(DEFAULT_DB_PATH))

VIEWS = [
    "Overview",
    "Ingest",
    "Knowledge Graph",
    "Paper Explorer",
    "Cross-Literature",
    "Figure Source Lookup",
]
view = st.sidebar.radio("View", VIEWS, label_visibility="collapsed")

store = get_store(db_path)
overview = db_overview(db_path, _token())
st.sidebar.divider()
st.sidebar.metric("Papers", overview["papers"])
st.sidebar.metric("Figures", overview["figures"])


# =============================================================== Overview


def view_overview() -> None:
    st.title("Overview")
    st.caption("What is currently in the knowledge base.")

    if overview["papers"] == 0:
        ui.empty_state(
            "The database is empty. Ingest some papers to get started.",
            hint='python scripts/run_arxiv_ingest.py "graph rag knowledge graph" --max 30',
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", overview["papers"])
    c2.metric("Figures", overview["figures"])
    c3.metric("Research areas", overview["areas"])
    c4.metric("Concepts", overview["concepts"])

    st.subheader("Papers by source")
    sources = overview["sources"]
    if sources:
        st.bar_chart(sources)

    st.subheader("Largest research areas")
    areas = research_areas(db_path, _token())[:15]
    if areas:
        st.dataframe(
            [{"research area": a["name"], "papers": a["count"], "from": a["source"]}
             for a in areas],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Pipeline status")
    steps = [
        ("Papers ingested", overview["papers"] > 0,
         'python scripts/run_arxiv_ingest.py "<query>" --max 30'),
        ("Embeddings + concepts", overview["concepts"] > 0,
         "python scripts/run_embed_papers.py --cluster kmeans"),
        ("Figures extracted", overview["figures"] > 0,
         "python scripts/run_extract_figures.py"),
    ]
    for label, done, cmd in steps:
        icon = ":white_check_mark:" if done else ":hourglass:"
        st.markdown(f"{icon} **{label}**")
        if not done:
            st.code(cmd, language="bash")


# ================================================================ Ingest


def view_ingest() -> None:
    st.title("Ingest")
    st.caption(
        "Add papers live via the Semantic Scholar API. For browser-based "
        "ArXiv scraping use the `run_arxiv_ingest.py` CLI — Playwright is "
        "better run outside the Streamlit process."
    )

    query = st.text_input("Search query", placeholder="e.g. graph retrieval augmented generation")
    max_results = st.slider("Max papers", min_value=5, max_value=100, value=25, step=5)

    if st.button("Ingest from Semantic Scholar", type="primary", disabled=not query):
        from xuanzhi.ingest import SemanticScholarSource

        async def _run() -> int:
            source = SemanticScholarSource()
            n = 0
            async for paper in source.search(query, max_results=max_results):
                store.upsert_paper(paper)
                n += 1
            return n

        with st.spinner(f"Querying Semantic Scholar for '{query}'…"):
            try:
                count = asyncio.run(_run())
            except Exception as e:  # noqa: BLE001
                st.error(f"Ingest failed: {e}")
                return
        st.success(f"Ingested {count} papers.")
        _bump_token()
        st.rerun()

    st.divider()
    st.subheader("Full pipeline (run in a terminal)")
    st.code(
        "python scripts/run_arxiv_ingest.py \"graph rag\" --max 50\n"
        "python scripts/run_embed_papers.py --cluster kmeans\n"
        "python scripts/run_summarise.py --limit 30\n"
        "python scripts/run_extract_figures.py --limit 20",
        language="bash",
    )


# ======================================================= Knowledge Graph


def view_graph() -> None:
    st.title("Knowledge Graph")
    st.caption("How the papers in the corpus connect to each other.")

    if overview["papers"] < 2:
        ui.empty_state(
            "Need at least two papers to draw a graph.",
            hint='python scripts/run_arxiv_ingest.py "<query>" --max 30',
        )
        return

    from xuanzhi.graph import build_paper_graph
    from xuanzhi.schema import EdgeType

    with st.sidebar:
        st.subheader("Graph filters")
        edge_choices = st.multiselect(
            "Edge types",
            options=[
                "shared_concept",
                "shared_author",
                "shared_area",
                "citation",
                "similar_embedding",
            ],
            default=["shared_concept", "shared_author"],
        )
        max_nodes = st.slider("Max papers (most-cited kept)", 20, 600, 200, step=20)
        min_year = st.number_input("Min year (0 = no filter)", value=0, step=1)

    edge_types = tuple(EdgeType(e) for e in edge_choices) or (EdgeType.SHARED_CONCEPT,)

    with st.spinner("Building graph…"):
        graph = build_paper_graph(
            store,
            edge_types=edge_types,
            min_year=int(min_year) or None,
            max_nodes=max_nodes,
        )

    c1, c2 = st.columns(2)
    c1.metric("Nodes", graph.number_of_nodes())
    c2.metric("Edges", graph.number_of_edges())
    ui.edge_legend()

    if graph.number_of_edges() == 0:
        st.info(
            "No edges with the current filters. Try enabling more edge types, "
            "or run `run_embed_papers.py --cluster kmeans` so papers share concepts."
        )

    clicked = ui.render_graph(graph, height=620)
    if clicked:
        paper = store.get_paper(clicked)
        if paper:
            st.divider()
            st.subheader("Selected paper")
            ui.paper_card(paper, expanded=True)


# ========================================================= Paper Explorer


def view_explorer() -> None:
    st.title("Paper Explorer")
    st.caption("Search the corpus; inspect summaries, figures and concepts.")

    papers = list_papers(db_path, _token())
    if not papers:
        ui.empty_state(
            "No papers yet.",
            hint='python scripts/run_arxiv_ingest.py "<query>" --max 30',
        )
        return

    search = st.text_input("Filter by title / author", placeholder="type to filter…")
    filtered = [
        p
        for p in papers
        if not search
        or search.lower() in (p["title"] or "").lower()
        or search.lower() in (p["authors"] or "").lower()
    ]
    st.caption(f"{len(filtered)} of {len(papers)} papers")
    if not filtered:
        st.warning("No papers match that filter.")
        return

    labels = {
        f"{p['title'][:80]}  ·  {p['year'] or '—'}": p["id"] for p in filtered
    }
    choice = st.selectbox("Paper", list(labels.keys()))
    paper = store.get_paper(labels[choice])
    if paper is None:
        st.error("Paper not found.")
        return

    st.divider()
    ui.paper_card(paper, expanded=True)

    tab_sum, tab_fig, tab_concepts, tab_related = st.tabs(
        ["Summaries", "Figures", "Concepts", "Related papers"]
    )
    with tab_sum:
        ui.summaries_block(store, paper)
    with tab_fig:
        ui.figure_grid(store, paper)
    with tab_concepts:
        concepts = store.get_concepts_for_paper(paper.id)
        if concepts:
            st.write(
                " ".join(f"`{c.name}`" for c, _ in concepts)
            )
        else:
            ui.empty_state(
                "No concepts attached yet.",
                hint="python scripts/run_embed_papers.py --cluster kmeans",
            )
    with tab_related:
        _related_papers(paper)


def _related_papers(paper) -> None:
    """Embedding-nearest neighbours of the selected paper, if embeddings exist."""
    try:
        from xuanzhi.nlp.embeddings import EmbeddingMatrix
    except ImportError:
        st.info("Install the NLP dependencies to see related papers.")
        return

    model = "sentence-transformers/all-MiniLM-L6-v2"
    matrix = EmbeddingMatrix.from_store(store, model=model)
    if len(matrix) == 0 or paper.id not in matrix.paper_ids:
        ui.empty_state(
            "No embeddings yet — related papers use embedding similarity.",
            hint="python scripts/run_embed_papers.py",
        )
        return
    idx = matrix.paper_ids.index(paper.id)
    for pid, score in matrix.cosine_similar(idx, top_k=5):
        related = store.get_paper(pid)
        if related:
            st.markdown(f"**{related.title}**  ·  similarity {score:.3f}")
            st.caption(
                f"{related.year or '—'} · {related.source.value}"
            )


# ======================================================= Cross-Literature


def view_cross() -> None:
    st.title("Cross-Literature")
    st.caption(
        "Pick two research areas — Xuanzhi surfaces the concepts they share "
        "and the papers that bridge them."
    )

    areas = research_areas(db_path, _token())
    areas = [a for a in areas if a["count"] > 0]
    if len(areas) < 2:
        ui.empty_state(
            "Need at least two research areas with papers.",
            hint='python scripts/run_arxiv_ingest.py "<query>" --max 50',
        )
        return

    labels = {f"{a['name']}  ({a['count']} papers)": a["id"] for a in areas}
    names = list(labels.keys())
    c1, c2 = st.columns(2)
    pick_a = c1.selectbox("Research area A", names, index=0)
    pick_b = c2.selectbox("Research area B", names, index=min(1, len(names) - 1))

    if labels[pick_a] == labels[pick_b]:
        st.warning("Choose two different areas.")
        return

    from xuanzhi.graph import cross_literature

    result = cross_literature(store, labels[pick_a], labels[pick_b])
    if result.is_empty:
        st.warning("One of the areas has no papers.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Papers in {result.area_a_name}", len(result.papers_a))
    m2.metric(f"Papers in {result.area_b_name}", len(result.papers_b))
    m3.metric("Bridging papers", len(result.bridging_papers))

    st.subheader("Shared concepts")
    if result.shared_concepts:
        st.dataframe(
            [{"concept": name, "frequency": freq}
             for name, freq in result.shared_concepts],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No shared concepts found. Run "
            "`run_embed_papers.py --cluster kmeans` so papers carry concepts."
        )

    st.subheader("Bridging papers")
    if result.bridging_papers:
        for pid in result.bridging_papers[:25]:
            bp = store.get_paper(pid)
            if bp:
                with st.container(border=True):
                    ui.paper_card(bp)
    else:
        st.info("No bridging papers with the current concept data.")


# =================================================== Figure Source Lookup


def view_figure_lookup() -> None:
    st.title("Figure Source Lookup")
    st.caption(
        "Upload a figure you want to reuse — Xuanzhi finds the source paper "
        "so you can cite it correctly."
    )

    if overview["figures"] == 0:
        ui.empty_state(
            "No figures indexed yet.",
            hint="python scripts/run_extract_figures.py",
        )
        return

    uploaded = st.file_uploader(
        "Figure image", type=["png", "jpg", "jpeg", "webp", "bmp"]
    )
    top_k = st.slider("Results", 1, 10, 5)

    if uploaded is None:
        st.info("Upload an image to search.")
        return

    # Persist the upload to a temp file for the CLIP encoder.
    import tempfile

    suffix = "." + uploaded.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    left, right = st.columns([1, 2])
    with left:
        st.image(tmp_path, caption="Your query image", use_container_width=True)

    with right:
        try:
            from xuanzhi.app.data import get_figure_index

            index = get_figure_index(db_path, _token())
            with st.spinner("Searching the figure index…"):
                matches = index.search(store, tmp_path, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            st.error(f"Lookup failed: {e}")
            return

        if not matches:
            st.warning("No similar figures found in the index.")
            return

        for rank, m in enumerate(matches, 1):
            with st.container(border=True):
                st.markdown(f"**#{rank} · similarity {m.similarity:.3f}**")
                fc1, fc2 = st.columns([1, 2])
                with fc1:
                    from pathlib import Path as _P

                    if _P(m.figure.image_path).exists():
                        st.image(m.figure.image_path, use_container_width=True)
                with fc2:
                    st.write(m.citation_line())
                    if m.figure.caption:
                        st.caption(m.figure.caption[:200])
                    if m.paper and m.paper.url:
                        st.markdown(f"[Source paper]({m.paper.url})")


# ================================================================= router

_ROUTER = {
    "Overview": view_overview,
    "Ingest": view_ingest,
    "Knowledge Graph": view_graph,
    "Paper Explorer": view_explorer,
    "Cross-Literature": view_cross,
    "Figure Source Lookup": view_figure_lookup,
}

_ROUTER[view]()
