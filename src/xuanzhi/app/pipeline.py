"""In-process pipeline runners for the Streamlit app.

Each function here wraps one of the ``scripts/`` CLIs as an importable
call, so the app can offer a button instead of a terminal command. They
take an already-open :class:`~xuanzhi.db.Store` plus the same knobs the
CLI exposes, do the work synchronously, and return a short
human-readable summary string. Failures raise — the caller shows
``st.error``.

The bodies mirror the step functions in ``scripts/run_demo.py``; keeping
them here means the app and the demo runner share one implementation and
the views stay declarative.

Heavy imports (sentence-transformers, torch, PyMuPDF, Playwright) are
done lazily inside each function so the app starts even when an optional
dependency is missing — only the button that needs it will fail.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

from xuanzhi.db import Store

# …/src/xuanzhi/app/pipeline.py  ->  repo root is four parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data"

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------- 1. ingest


def ingest_arxiv(
    store: Store, query: str, max_papers: int, *, headless: bool = True
) -> str:
    """Scrape ArXiv via Playwright and upsert the papers (run_arxiv_ingest.py)."""
    from xuanzhi.ingest import ArxivPlaywrightSource

    async def _run() -> int:
        source = ArxivPlaywrightSource(headless=headless)
        n = 0
        async for paper in source.search(query, max_results=max_papers):
            store.upsert_paper(paper)
            n += 1
        return n

    n = asyncio.run(_run())
    return f"Ingested {n} papers from ArXiv for '{query}'."


# ---------------------------------------------------- 2. embed + cluster


def embed_and_cluster(
    store: Store,
    *,
    model: str = DEFAULT_EMBED_MODEL,
    only_missing: bool = True,
    cluster: bool = True,
    method: str = "kmeans",
    k: int | None = None,
    min_cluster_size: int = 5,
) -> str:
    """Embed papers and optionally cluster them into concepts (run_embed_papers.py).

    Mirrors the CLI flags: ``model`` (--model), ``only_missing`` (the inverse
    of --all), ``method`` (--cluster kmeans|hdbscan), ``k`` (--k) and
    ``min_cluster_size`` (--min-cluster-size).
    """
    from xuanzhi.nlp import Embedder, cluster_embeddings, derive_concepts_from_clusters
    from xuanzhi.nlp.embeddings import EmbeddingMatrix, embed_corpus

    embedder = Embedder(model_name=model)
    n = embed_corpus(store, embedder, only_missing=only_missing)
    if not cluster:
        return f"Embedded {n} papers with {model}."

    matrix = EmbeddingMatrix.from_store(store, model=model)
    if len(matrix) < 2:
        return f"Embedded {n} papers — too few to cluster."
    labels = cluster_embeddings(
        matrix, method=method, k=k, min_cluster_size=min_cluster_size
    )
    concepts = derive_concepts_from_clusters(store, matrix, labels, persist=True)
    return (
        f"Embedded {n} papers; clustered into {len(concepts)} concepts "
        f"via {method}."
    )


# ------------------------------------------------------------ 3. summarise


DEFAULT_HF_SUMMARISER = "facebook/bart-large-cnn"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def summarise(
    store: Store,
    *,
    limit: int | None = None,
    hf_model: str = DEFAULT_HF_SUMMARISER,
    max_words: int = 80,
    use_openai: bool = False,
    openai_model: str = DEFAULT_OPENAI_MODEL,
) -> str:
    """Write a Summary row per paper with the HF model (run_summarise.py).

    Mirrors the CLI flags: ``limit`` (--limit), ``hf_model`` (--hf-model),
    ``max_words`` (--max-words), ``use_openai`` (--openai) and
    ``openai_model`` (--openai-model).
    """
    from xuanzhi.nlp import HFSummariser, OpenAISummariser

    papers = list(store.iter_papers(limit=limit))
    if not papers:
        return "DB is empty — ingest papers first."

    summarisers = [HFSummariser(model_name=hf_model)]
    if use_openai:
        summarisers.append(OpenAISummariser(model_name=openai_model))

    for paper in papers:
        for summariser in summarisers:
            summariser.summarise_to_store(store, paper, max_words=max_words)
    return f"Summarised {len(papers)} papers with {len(summarisers)} model(s)."


# --------------------------------------------------- 4. compare summarisers


def compare(
    store: Store,
    *,
    hf_models: list[str] | None = None,
    limit: int = 25,
    max_words: int = 80,
    use_openai: bool = False,
    openai_model: str = DEFAULT_OPENAI_MODEL,
) -> str:
    """Run the summariser comparison harness and write a report
    (run_compare_summarisers.py).

    Mirrors the CLI flags: ``hf_models`` (--hf-models, one or more ids),
    ``limit`` (--limit), ``max_words`` (--max-words), ``use_openai``
    (--openai) and ``openai_model`` (--openai-model). The report is written
    to ``data/outputs/`` — the CLI's --out default.
    """
    from xuanzhi.nlp import HFSummariser, OpenAISummariser
    from xuanzhi.nlp.compare import compare_summarisers, write_report

    papers = list(store.iter_papers(limit=limit))
    if not papers:
        return "DB is empty — ingest papers first."

    hf_models = hf_models or [DEFAULT_HF_SUMMARISER]
    summarisers = [HFSummariser(model_name=m) for m in hf_models]
    if use_openai:
        summarisers.append(OpenAISummariser(model_name=openai_model))

    df = compare_summarisers(summarisers, papers, max_words=max_words)
    if df.empty:
        return "No rows produced — every summariser errored."

    csv_path, json_path = write_report(df, _DATA_DIR / "outputs")
    return (
        f"Compared {len(summarisers)} summariser(s) over {len(papers)} papers "
        f"→ {csv_path.name} + {json_path.name}"
    )


# -------------------------------------------------------------- 5. figures


def extract_figures(
    store: Store,
    *,
    limit: int | None = None,
    classify: bool = True,
    build_index: bool = True,
    overwrite_pdf: bool = False,
) -> str:
    """Download PDFs, extract + classify figures, build the CLIP index
    (run_extract_figures.py).

    Mirrors the CLI flags: ``limit`` (--limit), ``classify`` (the inverse of
    --no-classify), ``build_index`` (the inverse of --no-index) and
    ``overwrite_pdf`` (--overwrite-pdf). PDFs and figures land in the CLI's
    default ``data/pdfs`` and ``data/figures`` directories.
    """
    from xuanzhi.cv import (
        FigureClassifier,
        FigureIndex,
        download_pdf,
        extract_figures as _extract,
    )

    pdf_dir = _DATA_DIR / "pdfs"
    figures_dir = _DATA_DIR / "figures"

    papers = list(store.iter_papers(limit=limit))
    if not papers:
        return "DB is empty — ingest papers first."

    classifier = FigureClassifier() if classify else None
    total = 0
    papers_with_pdf = 0
    type_counts: Counter[str] = Counter()

    for paper in papers:
        if not paper.pdf_url:
            continue
        pdf_path = download_pdf(
            str(paper.pdf_url), paper.id, pdf_dir, overwrite=overwrite_pdf
        )
        if pdf_path is None:
            continue
        papers_with_pdf += 1
        figures = _extract(pdf_path, paper.id, figures_dir)
        for fig in figures:
            if classifier is not None:
                fig = classifier.classify_figure(fig)
            store.add_figure(fig)
            type_counts[fig.figure_type.value] += 1
        total += len(figures)

    msg = f"Extracted {total} figures from {papers_with_pdf} papers."
    if type_counts:
        breakdown = ", ".join(
            f"{count} {fig_type}" for fig_type, count in type_counts.most_common()
        )
        msg += f" Types: {breakdown}."
    if build_index and total:
        n_indexed = FigureIndex().build_from_store(store, only_missing=True)
        msg += f" Indexed {n_indexed} figures into figure_embeddings."
    return msg
