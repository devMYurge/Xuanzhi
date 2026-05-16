"""Cached data-access layer for the Streamlit app.

Streamlit re-runs the whole script on every interaction, so anything
that touches the DB or loads a model goes through ``st.cache_*`` here.
Keeping all caching in one module means the views stay declarative and
we have a single place to reason about staleness.

Cache strategy
--------------
* ``@st.cache_resource`` for handles that should live for the whole
  session — the Store, the CLIP figure index.
* ``@st.cache_data`` for query results — invalidated by the ``_token``
  argument, which callers bump after an ingest so views refresh.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from xuanzhi.db import Store

# Resolve the default DB path relative to the repo root (…/src/xuanzhi/app/data.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = _REPO_ROOT / "data" / "xuanzhi.db"


@st.cache_resource(show_spinner=False)
def get_store(db_path: str) -> Store:
    """One Store per DB path for the whole session."""
    return Store(db_path)


@st.cache_data(show_spinner=False)
def db_overview(db_path: str, _token: int) -> dict:
    """Headline counts for the Overview dashboard."""
    store = get_store(db_path)
    return {
        "papers": store.count_papers(),
        "figures": store.count_figures(),
        "sources": store.source_counts(),
        "areas": len(store.list_research_areas()),
        "concepts": len(store.list_concepts()),
    }


@st.cache_data(show_spinner=False)
def list_papers(db_path: str, _token: int, limit: int | None = None) -> list[dict]:
    """Lightweight paper rows for tables / selectboxes (no nested objects)."""
    store = get_store(db_path)
    rows = []
    for p in store.iter_papers(limit=limit):
        rows.append(
            {
                "id": p.id,
                "title": p.title,
                "year": p.year,
                "source": p.source.value,
                "citation_count": p.citation_count,
                "authors": ", ".join(a.name for a in p.authors[:4]),
                "areas": ", ".join(a.name for a in p.research_areas),
            }
        )
    return rows


@st.cache_data(show_spinner=False)
def research_areas(db_path: str, _token: int) -> list[dict]:
    """Research areas with paper counts, for the cross-literature pickers."""
    store = get_store(db_path)
    counts = store.area_paper_counts()
    out = []
    for a in store.list_research_areas():
        out.append(
            {"id": a.id, "name": a.name, "source": a.source, "count": counts.get(a.id, 0)}
        )
    out.sort(key=lambda r: r["count"], reverse=True)
    return out


@st.cache_resource(show_spinner=False)
def get_figure_index(db_path: str, _token: int):
    """Load (and fit) the CLIP figure index once per session.

    Imported lazily so the app starts even if torch / sentence-transformers
    are not installed yet — only the Figure Source Lookup view needs them.
    """
    from xuanzhi.cv import FigureIndex

    store = get_store(db_path)
    return FigureIndex().load(store)
