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
    "Knowledge Graph",
    "Collection",
    "Overview",
    "Ingest",
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
    st.caption("Explore your corpus — click a node to open its paper.")

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

    # Collect authors + abstract for the in-graph click card.
    paper_extras: dict[str, dict] = {}
    for nid in graph.nodes():
        p = store.get_paper(nid)
        if p:
            paper_extras[nid] = {
                "authors": ", ".join(a.name for a in p.authors[:5]),
                "abstract": (p.abstract or "")[:360],
            }

    selected_id = ui.render_graph(graph, height=580, paper_extras=paper_extras)
    if selected_id:
        paper = store.get_paper(selected_id)
        if paper:
            st.divider()
            ui.paper_detail_panel(store, paper)


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

# ---- Citation formatters ------------------------------------------------

def _last_first(name: str) -> str:
    """'John Smith' → 'Smith, J.'  (best-effort; works for most Western names)."""
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0]
    initials = ". ".join(p[0].upper() for p in parts[:-1]) + "."
    return f"{parts[-1]}, {initials}"


def _format_citation(paper, style: str) -> str:
    """Return a formatted citation string for *paper* in the requested style."""
    if paper is None:
        return "(source paper not found in database)"

    authors = paper.authors or []
    title = paper.title or "Untitled"
    year = str(paper.year) if paper.year else "n.d."
    doi = paper.doi or ""
    url = paper.url or ""

    # Prefer arXiv URL when we have the ID
    arxiv_id = (paper.raw_metadata or {}).get("externalIds", {}).get("ArXiv", "")
    if arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"

    doi_str = f"https://doi.org/{doi}" if doi else url
    venue = paper.venue or (f"arXiv preprint arXiv:{arxiv_id}" if arxiv_id else "")

    names = [a.name for a in authors]

    if style == "APA":
        # Last, F., Last, F., & Last, F. (year). Title. Venue. doi
        if not names:
            author_str = "Unknown Author"
        elif len(names) == 1:
            author_str = _last_first(names[0])
        elif len(names) == 2:
            author_str = f"{_last_first(names[0])}, & {_last_first(names[1])}"
        elif len(names) <= 7:
            parts = [_last_first(n) for n in names[:-1]]
            author_str = ", ".join(parts) + f", & {_last_first(names[-1])}"
        else:
            parts = [_last_first(n) for n in names[:6]]
            author_str = ", ".join(parts) + f", ... {_last_first(names[-1])}"
        venue_part = f" {venue}." if venue else ""
        doi_part = f" {doi_str}" if doi_str else ""
        return f"{author_str} ({year}). {title}.{venue_part}{doi_part}"

    elif style == "MLA":
        # Last, First, et al. "Title." Venue, year. URL.
        if not names:
            author_str = "Unknown Author"
        elif len(names) == 1:
            parts = names[0].strip().split()
            author_str = f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else names[0]
        elif len(names) == 2:
            p0 = names[0].strip().split()
            first = f"{p0[-1]}, {' '.join(p0[:-1])}" if len(p0) > 1 else names[0]
            author_str = f"{first}, and {names[1]}"
        else:
            p0 = names[0].strip().split()
            first = f"{p0[-1]}, {' '.join(p0[:-1])}" if len(p0) > 1 else names[0]
            author_str = f"{first}, et al."
        venue_part = f" {venue}," if venue else ""
        url_part = f" {url}." if url else "."
        return f'{author_str} "{title}."{venue_part} {year}.{url_part}'

    elif style == "Harvard":
        # Last, F., Last, F. and Last, F. (year) 'Title', Venue. doi
        if not names:
            author_str = "Unknown Author"
        elif len(names) == 1:
            author_str = _last_first(names[0])
        elif len(names) == 2:
            author_str = f"{_last_first(names[0])} and {_last_first(names[1])}"
        elif len(names) <= 3:
            parts = [_last_first(n) for n in names[:-1]]
            author_str = ", ".join(parts) + f" and {_last_first(names[-1])}"
        else:
            author_str = f"{_last_first(names[0])} et al."
        venue_part = f", {venue}" if venue else ""
        doi_part = f". Available at: {doi_str}" if doi_str else ""
        return f"{author_str} ({year}) '{title}'{venue_part}{doi_part}"

    elif style == "Chicago":
        # Last, First, First Last, and First Last. "Title." Venue (year). doi.
        if not names:
            author_str = "Unknown Author"
        elif len(names) == 1:
            parts = names[0].strip().split()
            author_str = f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else names[0]
        elif len(names) == 2:
            p0 = names[0].strip().split()
            first = f"{p0[-1]}, {' '.join(p0[:-1])}" if len(p0) > 1 else names[0]
            author_str = f"{first}, and {names[1]}"
        elif len(names) <= 3:
            p0 = names[0].strip().split()
            first = f"{p0[-1]}, {' '.join(p0[:-1])}" if len(p0) > 1 else names[0]
            middle = ", ".join(names[1:-1])
            author_str = f"{first}, {middle}, and {names[-1]}"
        else:
            p0 = names[0].strip().split()
            first = f"{p0[-1]}, {' '.join(p0[:-1])}" if len(p0) > 1 else names[0]
            author_str = f"{first} et al."
        venue_part = f" {venue}" if venue else ""
        doi_part = f" {doi_str}." if doi_str else ""
        return f'{author_str}. "{title}."{venue_part} ({year}).{doi_part}'

    # Fallback — plain
    plain_authors = ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")
    return f"{plain_authors} ({year}). {title}."


def _normalise_s2_id(raw: str) -> str:
    """Normalise a user-typed arXiv ID or DOI to Semantic Scholar format."""
    import re
    s = raw.strip()
    if re.match(r"^10\.", s):
        return f"DOI:{s}"
    if s.upper().startswith("DOI:"):
        return s
    clean = re.sub(r"^(arxiv:?\s*)", "", s, flags=re.IGNORECASE)
    return f"ARXIV:{clean}"


def _ingest_uploaded_paper(
    store,
    pdf_bytes: bytes,
    paper_id_raw: str,
    manual_title: str,
    manual_authors: str,
    manual_year: int | None,
) -> tuple[int, str]:
    """Save a user-uploaded PDF, extract + classify + embed its figures.

    Returns (n_figures_added, paper_title).
    """
    import asyncio
    from pathlib import Path as _P

    from xuanzhi.cv.classify import FigureClassifier
    from xuanzhi.cv.figures import extract_figures
    from xuanzhi.cv.index import FigureIndex
    from xuanzhi.schema import Author, Paper, Source

    repo_root = DEFAULT_DB_PATH.parent.parent
    pdfs_dir = repo_root / "data" / "pdfs"
    figures_dir = repo_root / "data" / "figures"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    # --- fetch or build paper record ---
    if paper_id_raw.strip():
        s2_id = _normalise_s2_id(paper_id_raw)
        from xuanzhi.ingest import SemanticScholarSource

        async def _fetch():
            return await SemanticScholarSource().enrich(s2_id)

        paper = asyncio.run(_fetch())
        if paper is None:
            raise ValueError(f"No paper found for '{s2_id}'. Double-check the ID.")
    else:
        if not manual_title.strip():
            raise ValueError("Provide an ArXiv/DOI ID, or fill in the Title field.")
        authors = [
            Author.from_name(n.strip())
            for n in manual_authors.split(",")
            if n.strip()
        ]
        paper = Paper(
            id=Paper.build_id(Source.MANUAL, manual_title.strip()),
            source=Source.MANUAL,
            source_id=manual_title.strip(),
            title=manual_title.strip(),
            authors=authors,
            year=int(manual_year) if manual_year else None,
        )

    store.upsert_paper(paper)

    # --- save PDF ---
    pdf_path = pdfs_dir / f"{paper.id}.pdf"
    pdf_path.write_bytes(pdf_bytes)

    # --- extract, classify, persist figures ---
    figs = extract_figures(pdf_path, paper.id, figures_dir)
    classifier = FigureClassifier()
    for fig in figs:
        fig = classifier.classify_figure(fig)
        store.add_figure(fig)

    # --- embed only the new figures (skips already-embedded ones) ---
    FigureIndex().build_from_store(store, only_missing=True)

    return len(figs), paper.title


def view_figure_lookup() -> None:
    st.title("Figure Source Lookup")
    st.caption(
        "Upload a figure and Xuanzhi finds the source paper so you can cite it. "
        "If the paper isn't in the index yet, attach its PDF too — it will be "
        "ingested automatically before the search runs."
    )

    # ---- Inputs ----------------------------------------------------------
    fig_col, pdf_col = st.columns(2)

    with fig_col:
        st.markdown("**Figure image** *(required)*")
        uploaded_fig = st.file_uploader(
            "Figure image", type=["png", "jpg", "jpeg", "webp", "bmp"],
            label_visibility="collapsed", key="fig_image_upload",
        )

    with pdf_col:
        st.markdown("**Source PDF** *(optional — attach if paper is not yet indexed)*")
        uploaded_pdf = st.file_uploader(
            "Source PDF", type=["pdf"],
            label_visibility="collapsed", key="fig_pdf_upload",
        )

    # PDF metadata fields — only shown when a PDF is attached
    paper_id_raw = ""
    manual_title = manual_authors = ""
    manual_year = 2025
    if uploaded_pdf is not None:
        paper_id_raw = st.text_input(
            "ArXiv ID or DOI (recommended — fetches metadata automatically)",
            placeholder="e.g. 2401.12345  or  DOI:10.1145/3626246",
            key="fig_paper_id",
        )
        st.caption("Or enter metadata manually if you don't have an ID:")
        mc1, mc2, mc3 = st.columns([3, 2, 1])
        manual_title = mc1.text_input("Title", key="fig_manual_title")
        manual_authors = mc2.text_input("Authors (comma-separated)", key="fig_manual_authors")
        manual_year = mc3.number_input(
            "Year", min_value=1900, max_value=2030, value=2025, step=1,
            key="fig_manual_year",
        )

    top_k = st.slider("Results to show", 1, 10, 5)

    search_clicked = st.button(
        "Search", type="primary", disabled=uploaded_fig is None,
    )

    if not search_clicked:
        if uploaded_fig is None:
            st.info("Upload a figure image to get started.")
        return

    # ---- Run (ingest PDF if provided, then search) -----------------------
    import tempfile
    from pathlib import Path as _P

    suffix = "." + uploaded_fig.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_fig.getbuffer())
        tmp_path = tmp.name

    if uploaded_pdf is not None:
        with st.spinner("Ingesting PDF — extracting and indexing figures…"):
            try:
                n_figs, title = _ingest_uploaded_paper(
                    store,
                    pdf_bytes=uploaded_pdf.getvalue(),
                    paper_id_raw=paper_id_raw,
                    manual_title=manual_title,
                    manual_authors=manual_authors,
                    manual_year=int(manual_year) if manual_year else None,
                )
                st.success(f"Added **{title}** — {n_figs} figures indexed.")
                _bump_token()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not ingest PDF: {e}")
                return

    if overview["figures"] == 0 and uploaded_pdf is None:
        ui.empty_state(
            "No figures in the index yet. Attach a source PDF above, or run:",
            hint="python scripts/run_extract_figures.py",
        )
        return

    left, right = st.columns([1, 2])
    with left:
        st.image(tmp_path, caption="Your figure", use_container_width=True)

    with right:
        try:
            from xuanzhi.app.data import get_figure_index

            index = get_figure_index(db_path, _token())
            with st.spinner("Searching…"):
                matches = index.search(store, tmp_path, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            st.error(f"Search failed: {e}")
            return

        if not matches:
            st.warning(
                "No similar figures found. Try attaching the source PDF so "
                "Xuanzhi can index it."
            )
            return

        cite_style = st.selectbox(
            "Citation format",
            ["APA", "MLA", "Harvard", "Chicago"],
            index=0,
        )

        for rank, m in enumerate(matches, 1):
            with st.container(border=True):
                st.markdown(f"**#{rank} · similarity {m.similarity:.3f}**")
                fc1, fc2 = st.columns([1, 2])
                with fc1:
                    if _P(m.figure.image_path).exists():
                        st.image(m.figure.image_path, use_container_width=True)
                with fc2:
                    if m.figure.caption:
                        st.caption(m.figure.caption[:200])
                    citation = _format_citation(m.paper, cite_style)
                    st.code(citation, language=None)
                    if m.paper and m.paper.url:
                        arxiv_id = (m.paper.raw_metadata or {}).get("externalIds", {}).get("ArXiv", "")
                        link_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else m.paper.url
                        st.markdown(f"[Source paper]({link_url})")


# ============================================================== Collection


def view_collection() -> None:
    st.title("Collection")
    st.caption("Papers you have saved while browsing the graph.")

    collection: list = st.session_state.setdefault("collection", [])

    if not collection:
        st.info("Nothing saved yet. Click a node in the Knowledge Graph and hit ☆ Save.")
        return

    st.caption(f"{len(collection)} saved paper{'s' if len(collection) != 1 else ''}")
    if st.button("Clear all"):
        st.session_state["collection"] = []
        st.rerun()

    st.divider()
    for pid in list(collection):
        paper = store.get_paper(pid)
        if paper is None:
            continue
        with st.container(border=True):
            col_card, col_rm = st.columns([9, 1])
            with col_card:
                ui.paper_card(paper, expanded=False)
            with col_rm:
                if st.button("✕", key=f"rm_{pid}"):
                    collection.remove(pid)
                    st.rerun()


# ================================================================= router

_ROUTER = {
    "Knowledge Graph": view_graph,
    "Collection": view_collection,
    "Overview": view_overview,
    "Ingest": view_ingest,
    "Paper Explorer": view_explorer,
    "Cross-Literature": view_cross,
    "Figure Source Lookup": view_figure_lookup,
}

_ROUTER[view]()
