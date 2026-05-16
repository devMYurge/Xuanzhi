"""Summarise every paper in the DB with a HuggingFace model (and optionally
OpenAI), writing Summary rows.

Run:
    # HF only
    python scripts/run_summarise.py

    # HF + OpenAI (needs OPENAI_API_KEY)
    python scripts/run_summarise.py --openai

    # limit to N papers while testing
    python scripts/run_summarise.py --limit 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from xuanzhi.db import Store
from xuanzhi.nlp import HFSummariser, OpenAISummariser


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise papers into the DB.")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "xuanzhi.db")
    parser.add_argument(
        "--hf-model",
        default="facebook/bart-large-cnn",
        help="HuggingFace summarisation model id.",
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Also summarise with OpenAI (needs OPENAI_API_KEY).",
    )
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None, help="Cap papers processed.")
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

    summarisers = [HFSummariser(model_name=args.hf_model)]
    if args.openai:
        summarisers.append(OpenAISummariser(model_name=args.openai_model))

    for i, paper in enumerate(papers, 1):
        for summariser in summarisers:
            summary = summariser.summarise_to_store(
                store, paper, max_words=args.max_words
            )
            print(
                f"[{i:3d}/{len(papers)}] {summariser.model_name:24s} "
                f"{paper.title[:60]:60s} -> {summary.summary_text[:70]}"
            )

    print(f"\nDone. Summarised {len(papers)} papers with "
          f"{len(summarisers)} model(s).")


if __name__ == "__main__":
    main()
