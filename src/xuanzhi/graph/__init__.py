"""Graph layer: build a networkx knowledge graph over the paper database
and run the cross-literature queries on top of it.

This is where the pieces converge — ingest fills the DB, NLP attaches
concepts and embeddings, and this module turns all of that into the
navigable graph the Streamlit UI renders.
"""

from .build import (
    CrossLiteratureResult,
    build_paper_graph,
    cross_literature,
    persist_derived_edges,
)

__all__ = [
    "CrossLiteratureResult",
    "build_paper_graph",
    "cross_literature",
    "persist_derived_edges",
]
