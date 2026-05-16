"""Smoke-test the Playwright ArXiv ingest end-to-end.

Run:
    python scripts/run_arxiv_ingest.py "graph rag knowledge graph" --max 10

The script:
    1. Spins up the Playwright Chromium browser.
    2. Streams Paper objects out of ArXiv search results.
    3. Upserts each into data/xuanzhi.db.
    4. Prints a one-line summary per paper.

Useful for verifying the schema, the scraper, and the DB layer all line
up before plugging in the NLP / CV / Streamlit pieces.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make src/ importable when running from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from xuanzhi.db import Store
from xuanzhi.ingest import ArxivPlaywrightSource


async def amain(query: str, max_results: int, db_path: Path, headless: bool) -> None:
    store = Store(db_path)
    source = ArxivPlaywrightSource(headless=headless)
    n = 0
    async for paper in source.search(query, max_results=max_results):
        store.upsert_paper(paper)
        n += 1
        print(f"[{n:3d}] {paper.source_id:14s}  {paper.title[:90]}")
    print(f"\nIngested {n} papers. DB now holds {store.count_papers()} papers total.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test ArXiv Playwright ingest.")
    parser.add_argument("query", help="Search query passed to ArXiv.")
    parser.add_argument("--max", type=int, default=10, help="Max papers to ingest.")
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "data" / "xuanzhi.db",
        help="SQLite DB path.",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run Playwright non-headless so you can watch the scrape.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(
        amain(
            query=args.query,
            max_results=args.max,
            db_path=args.db,
            headless=not args.show_browser,
        )
    )


if __name__ == "__main__":
    main()
