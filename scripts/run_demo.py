"""One-shot demo runner — execute the whole Xuanzhi pipeline end-to-end.

Run:
    # default: 25 papers on the project's flagship topic
    python scripts/run_demo.py

    # quick smoke (10 papers, 5 figures, no OpenAI)
    python scripts/run_demo.py --preset quick

    # full demo for the slides (50 papers, 25 figures, HF + OpenAI compare)
    python scripts/run_demo.py --preset full --openai

    # custom
    python scripts/run_demo.py --query "vision transformers" --max 30

The script orchestrates the six existing CLIs in order, with consistent
section headers, per-step timing, and an end-of-run dashboard. Failures
in optional steps (figures, OpenAI comparison) are caught so the rest of
the pipeline finishes — you get *some* demo even if one piece breaks.

Pipeline:
    1. ArXiv ingest (Playwright)                   -> data/xuanzhi.db
    2. Embed papers (sentence-transformers)        -> paper_embeddings
    3. Cluster -> concepts (sklearn KMeans + TFIDF)
    4. Summarise with HF (bart-large-cnn)
    5. [optional --openai] HF vs OpenAI comparison -> data/outputs/*.{csv,json}
    6. [optional] Extract + classify + index figures (PyMuPDF + CLIP)
    7. Status dashboard + next-step hints
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ----- presets --------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "quick": {
        "max_papers": 10,
        "max_figures": 5,
        "summarise_limit": 10,
        "compare_limit": 10,
    },
    "default": {
        "max_papers": 25,
        "max_figures": 15,
        "summarise_limit": 25,
        "compare_limit": 25,
    },
    "full": {
        "max_papers": 50,
        "max_figures": 25,
        "summarise_limit": 30,
        "compare_limit": 25,
    },
}

# ----- timing + section helpers ---------------------------------------------


def banner(title: str, ch: str = "=") -> None:
    bar = ch * 70
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def step(name: str):
    """Context manager that times a pipeline step and reports the result."""
    return _StepTimer(name)


class _StepTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = 0.0
        self.elapsed: float = 0.0
        self.ok: bool = False
        self.error: str | None = None

    def __enter__(self):
        banner(self.name)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed = time.perf_counter() - self.t0
        if exc_type is None:
            self.ok = True
            print(f"\n[done] {self.name}  ({self.elapsed:.1f}s)", flush=True)
        else:
            self.ok = False
            self.error = f"{exc_type.__name__}: {exc_val}"
            print(
                f"\n[FAIL] {self.name}  ({self.elapsed:.1f}s) — {self.error}",
                flush=True,
            )
        # Swallow exceptions in optional steps; the caller decides via .ok.
        return True


# ----- the pipeline ---------------------------------------------------------


async def step_ingest(db_path: Path, query: str, max_papers: int, headless: bool) -> int:
    from xuanzhi.db import Store
    from xuanzhi.ingest import ArxivPlaywrightSource

    store = Store(db_path)
    source = ArxivPlaywrightSource(headless=headless)
    n = 0
    async for paper in source.search(query, max_results=max_papers):
        store.upsert_paper(paper)
        n += 1
        print(f"  [{n:3d}] {paper.source_id:14s}  {paper.title[:80]}")
    return n


def step_embed_and_cluster(db_path: Path, embed_model: str, k: int | None) -> dict:
    from xuanzhi.db import Store
    from xuanzhi.nlp import Embedder, cluster_embeddings, derive_concepts_from_clusters
    from xuanzhi.nlp.embeddings import EmbeddingMatrix, embed_corpus

    store = Store(db_path)
    embedder = Embedder(model_name=embed_model)
    n = embed_corpus(store, embedder, only_missing=True)
    print(f"  embedded {n} papers with {embed_model}")
    matrix = EmbeddingMatrix.from_store(store, model=embed_model)
    if len(matrix) < 2:
        print("  too few papers to cluster — skipping")
        return {"embedded": n, "clusters": 0}
    labels = cluster_embeddings(matrix, method="kmeans", k=k)
    concepts = derive_concepts_from_clusters(store, matrix, labels, persist=True)
    print(f"  clustered into {len(concepts)} concepts:")
    for cid, concept in sorted(concepts.items()):
        size = int((labels == cid).sum())
        print(f"    [{cid:2d}] ({size:3d})  {concept.name}")
    return {"embedded": n, "clusters": len(concepts)}


def step_summarise(db_path: Path, limit: int, max_words: int) -> int:
    from xuanzhi.db import Store
    from xuanzhi.nlp import HFSummariser

    store = Store(db_path)
    summariser = HFSummariser()
    papers = list(store.iter_papers(limit=limit))
    for i, paper in enumerate(papers, 1):
        summary = summariser.summarise_to_store(store, paper, max_words=max_words)
        print(f"  [{i:3d}/{len(papers)}] {paper.source_id:14s}  {summary.summary_text[:80]}")
    return len(papers)


def step_compare(
    db_path: Path,
    limit: int,
    max_words: int,
    use_openai: bool,
    out_dir: Path,
) -> dict:
    from xuanzhi.db import Store
    from xuanzhi.nlp import HFSummariser, OpenAISummariser
    from xuanzhi.nlp.compare import compare_summarisers, summarise_report, write_report

    store = Store(db_path)
    papers = list(store.iter_papers(limit=limit))
    summarisers = [HFSummariser()]
    if use_openai:
        summarisers.append(OpenAISummariser())
    print(f"  comparing {len(summarisers)} summariser(s) over {len(papers)} papers")
    df = compare_summarisers(summarisers, papers, max_words=max_words)
    if df.empty:
        print("  no rows produced — every summariser errored")
        return {"rows": 0}
    csv_path, json_path = write_report(df, out_dir)
    report = summarise_report(df)
    for model, stats in report["models"].items():
        print(
            f"  {model:26s}  lat {stats['mean_latency_s']:5.1f}s  "
            f"words {stats['mean_word_count']:5.1f}  "
            f"compress {stats['mean_compression_ratio']:.2f}  "
            f"rougeL {stats['mean_rouge_l']}"
        )
    print(f"  wrote {csv_path.name} + {json_path.name}")
    return {"rows": len(df), "csv": str(csv_path), "json": str(json_path)}


def step_figures(
    db_path: Path,
    pdf_dir: Path,
    figures_dir: Path,
    limit: int,
    classify: bool,
    build_index: bool,
) -> dict:
    from xuanzhi.cv import FigureClassifier, FigureIndex, download_pdf, extract_figures
    from xuanzhi.db import Store

    store = Store(db_path)
    papers = list(store.iter_papers(limit=limit))
    classifier = FigureClassifier() if classify else None

    total = 0
    type_counts: Counter[str] = Counter()
    n_with_pdf = 0
    for i, paper in enumerate(papers, 1):
        if not paper.pdf_url:
            continue
        pdf_path = download_pdf(str(paper.pdf_url), paper.id, pdf_dir)
        if pdf_path is None:
            print(f"  [{i:3d}/{len(papers)}] {paper.source_id:14s}  pdf failed")
            continue
        n_with_pdf += 1
        figures = extract_figures(pdf_path, paper.id, figures_dir)
        for fig in figures:
            if classifier is not None:
                fig = classifier.classify_figure(fig)
            store.add_figure(fig)
            type_counts[fig.figure_type.value] += 1
        total += len(figures)
        print(
            f"  [{i:3d}/{len(papers)}] {paper.source_id:14s}  "
            f"{len(figures):2d} figures  {paper.title[:54]}"
        )

    print(f"\n  extracted {total} figures from {n_with_pdf} papers")
    if type_counts:
        for fig_type, count in type_counts.most_common():
            print(f"    {fig_type:10s} {count}")

    n_indexed = 0
    if build_index and total:
        print("\n  building CLIP figure index…")
        index = FigureIndex()
        n_indexed = index.build_from_store(store, only_missing=True)
        print(f"  embedded {n_indexed} figures into figure_embeddings")

    return {
        "figures": total,
        "papers_with_pdf": n_with_pdf,
        "type_counts": dict(type_counts),
        "indexed": n_indexed,
    }


def final_dashboard(db_path: Path, steps: dict) -> None:
    from xuanzhi.db import Store

    store = Store(db_path)
    banner("Demo complete", "*")
    print(f"\nDatabase: {db_path}")
    print(f"  papers   : {store.count_papers()}")
    print(f"  figures  : {store.count_figures()}")
    print(f"  areas    : {len(store.list_research_areas())}")
    print(f"  concepts : {len(store.list_concepts())}")

    print("\nStep results:")
    for name, info in steps.items():
        status = info.get("status", "?")
        elapsed = info.get("elapsed", 0.0)
        line = f"  {status:6s} {name:32s} {elapsed:6.1f}s"
        details = info.get("details")
        if details:
            line += f"   {details}"
        print(line)

    print("\nNext:")
    print("  streamlit run streamlit_app.py")
    print("  then click through Overview -> Knowledge Graph -> Cross-Literature ->")
    print("  Figure Source Lookup; screenshot each for the slides.\n")


# ----- main -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole Xuanzhi pipeline.")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="default",
        help="Sizing preset: quick (~10), default (~25), full (~50 papers).",
    )
    parser.add_argument(
        "--query",
        default="graph retrieval augmented generation",
        help="ArXiv search query.",
    )
    parser.add_argument("--max", type=int, default=None, help="Override preset paper count.")
    parser.add_argument(
        "--db", type=Path, default=ROOT / "data" / "xuanzhi.db", help="SQLite DB path."
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=ROOT / "data" / "pdfs"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=ROOT / "data" / "figures"
    )
    parser.add_argument(
        "--outputs-dir", type=Path, default=ROOT / "data" / "outputs"
    )
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model id.",
    )
    parser.add_argument("--k", type=int, default=None, help="KMeans cluster count.")
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Include OpenAI in the comparison harness (needs OPENAI_API_KEY).",
    )
    parser.add_argument("--show-browser", action="store_true", help="Run Playwright non-headless.")
    parser.add_argument("--no-ingest", action="store_true", help="Skip ingest; reuse existing DB.")
    parser.add_argument("--no-figures", action="store_true", help="Skip the CV pipeline.")
    parser.add_argument("--no-compare", action="store_true", help="Skip the comparison harness.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    preset = PRESETS[args.preset].copy()
    if args.max is not None:
        preset["max_papers"] = args.max

    banner(
        f"Xuanzhi demo run · preset={args.preset} · query={args.query!r}",
        ch="#",
    )
    print(
        f"  papers: {preset['max_papers']}  figures: {preset['max_figures']}  "
        f"openai: {args.openai}"
    )

    steps: dict[str, dict] = {}

    def record(key: str, t: _StepTimer, details: str = "") -> None:
        """Always record the step result, even if the work raised."""
        steps[key] = {
            "status": "OK" if t.ok else "FAIL",
            "elapsed": t.elapsed,
            "details": details or (t.error or ""),
        }

    # ---- 1. ingest --------------------------------------------------------
    if args.no_ingest:
        print("\n(skipping ingest — reusing existing DB)")
    else:
        details = ""
        with step("1. ArXiv ingest (Playwright)") as t:
            n = asyncio.run(
                step_ingest(
                    args.db, args.query, preset["max_papers"], not args.show_browser
                )
            )
            details = f"{n} papers"
        record("ingest", t, details)
        if not t.ok:
            print("\nIngest failed — aborting. Try `--show-browser` to debug selectors.")
            final_dashboard(args.db, steps)
            return

    # ---- 2. embed + cluster ----------------------------------------------
    details = ""
    with step("2. Embed + cluster") as t:
        info = step_embed_and_cluster(args.db, args.embed_model, args.k)
        details = f"{info['embedded']} embedded, {info['clusters']} concepts"
    record("embed_cluster", t, details)

    # ---- 3. summarise -----------------------------------------------------
    details = ""
    with step("3. Summarise (HF bart-large-cnn)") as t:
        n = step_summarise(args.db, preset["summarise_limit"], args.max_words)
        details = f"{n} summaries"
    record("summarise", t, details)

    # ---- 4. compare -------------------------------------------------------
    if not args.no_compare:
        details = ""
        with step("4. Summariser comparison (HF" + (" + OpenAI" if args.openai else "") + ")") as t:
            info = step_compare(
                args.db,
                preset["compare_limit"],
                args.max_words,
                args.openai,
                args.outputs_dir,
            )
            details = f"{info.get('rows', 0)} rows"
            if "csv" in info:
                details += f" -> {Path(info['csv']).name}"
        record("compare", t, details)

    # ---- 5. figures -------------------------------------------------------
    if not args.no_figures:
        details = ""
        with step("5. Figures (extract + classify + index)") as t:
            info = step_figures(
                args.db,
                args.pdf_dir,
                args.figures_dir,
                preset["max_figures"],
                classify=True,
                build_index=True,
            )
            details = (
                f"{info['figures']} figs / {info['papers_with_pdf']} pdfs, "
                f"{info['indexed']} indexed"
            )
        record("figures", t, details)

    # ---- 6. dashboard -----------------------------------------------------
    final_dashboard(args.db, steps)


if __name__ == "__main__":
    main()
