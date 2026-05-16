"""Download PDFs, extract figures, classify them, and build the CLIP index.

Run:
    # full pipeline over every paper that has a PDF url
    python scripts/run_extract_figures.py

    # limit while testing, and watch the figure-type breakdown
    python scripts/run_extract_figures.py --limit 10

    # skip classification / skip index build
    python scripts/run_extract_figures.py --no-classify --no-index

Pipeline per paper:
    1. download_pdf()      -> data/pdfs/{paper_id}.pdf      (cached)
    2. extract_figures()   -> data/figures/{paper_id}/*.png + Figure rows
    3. FigureClassifier    -> figure_type tagged via CLIP zero-shot
    4. FigureIndex.build   -> CLIP image embeddings in figure_embeddings
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from xuanzhi.cv import FigureClassifier, FigureIndex, download_pdf, extract_figures
from xuanzhi.db import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract + classify + index figures.")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "xuanzhi.db")
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data" / "pdfs")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "data" / "figures")
    parser.add_argument("--limit", type=int, default=None, help="Cap papers processed.")
    parser.add_argument("--no-classify", action="store_true")
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument("--overwrite-pdf", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    store = Store(args.db)
    papers = list(store.iter_papers(limit=args.limit))
    if not papers:
        print("DB is empty — run scripts/run_arxiv_ingest.py first.")
        return

    classifier = None if args.no_classify else FigureClassifier()

    total_figs = 0
    type_counts: Counter[str] = Counter()
    papers_with_pdf = 0

    for i, paper in enumerate(papers, 1):
        if not paper.pdf_url:
            continue
        pdf_path = download_pdf(
            str(paper.pdf_url),
            paper.id,
            args.pdf_dir,
            overwrite=args.overwrite_pdf,
        )
        if pdf_path is None:
            print(f"[{i:3d}/{len(papers)}] {paper.source_id:14s}  PDF download failed")
            continue
        papers_with_pdf += 1

        figures = extract_figures(pdf_path, paper.id, args.figures_dir)
        for fig in figures:
            if classifier is not None:
                fig = classifier.classify_figure(fig)
            store.add_figure(fig)
            type_counts[fig.figure_type.value] += 1
        total_figs += len(figures)
        print(
            f"[{i:3d}/{len(papers)}] {paper.source_id:14s}  "
            f"{len(figures):2d} figures  {paper.title[:54]}"
        )

    print(
        f"\nExtracted {total_figs} figures from {papers_with_pdf} papers "
        f"({store.count_figures()} figures in DB total)."
    )
    if type_counts:
        print("Figure-type breakdown:")
        for fig_type, count in type_counts.most_common():
            print(f"  {fig_type:10s} {count}")

    if not args.no_index and total_figs:
        print("\nBuilding CLIP figure embedding index...")
        index = FigureIndex()
        n = index.build_from_store(store, only_missing=True)
        print(f"Embedded {n} figures into figure_embeddings.")


if __name__ == "__main__":
    main()
