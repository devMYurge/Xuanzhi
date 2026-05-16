"""Xuanzhi — academic-research cognition layer.

Module map:

    schema/     Unified Pydantic data model (Paper, Author, ResearchArea, Figure,
                Citation, Edge, ...). Single source of truth across the system.
    ingest/     Source-specific ingest modules (Playwright for sites that need a
                browser; REST for sites that have an API). All emit the unified
                schema.
    db/         SQLite storage layer mirroring the schema.
    nlp/        HuggingFace transformer pipelines (classification, summarisation,
                embeddings) + scikit-learn analytics. Frontier-model comparison
                lives here too.
    cv/         Figure extraction from PDFs + vision-model classification +
                similarity index for figure-to-source citation lookup.
    graph/      networkx graph construction over the database; connection
                discovery across literatures.
    app/        Streamlit UI (graph view, per-paper view, figure-source lookup).
    utils/      Shared helpers (logging, async pools, file io).
"""

__version__ = "0.1.0"
