"""Image-to-source lookup — the flagship demo feature.

Give it an image (a figure you want to cite) and it returns the source
paper(s) from the database, ranked by visual similarity.

Run:
    python scripts/run_figure_lookup.py path/to/some_figure.png
    python scripts/run_figure_lookup.py path/to/some_figure.png --top-k 3

Prerequisite: figures must already be extracted + embedded, i.e. run
    python scripts/run_extract_figures.py
first so figure_embeddings is populated.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from xuanzhi.cv import FigureIndex
from xuanzhi.db import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the source paper for an image.")
    parser.add_argument("image", type=Path, help="Path to the query image.")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "xuanzhi.db")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        return

    store = Store(args.db)
    index = FigureIndex().load(store)
    if len(index) == 0:
        print(
            "Figure index is empty — run scripts/run_extract_figures.py first "
            "so figure_embeddings is populated."
        )
        return

    matches = index.search(store, str(args.image), top_k=args.top_k)
    if not matches:
        print("No matches found.")
        return

    print(f"\nTop {len(matches)} source candidates for {args.image.name}:\n")
    for rank, m in enumerate(matches, 1):
        page = f"p.{m.figure.page_num}" if m.figure.page_num else "p.?"
        print(f"  {rank}. similarity {m.similarity:.3f}   [{page}]")
        print(f"     {m.citation_line()}")
        if m.figure.caption:
            print(f"     caption: {m.figure.caption[:100]}")
        if m.paper and m.paper.url:
            print(f"     {m.paper.url}")
        print()


if __name__ == "__main__":
    main()
