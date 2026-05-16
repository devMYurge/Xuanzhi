"""Run the HF-vs-OpenAI summariser comparison and write a report.

Run:
    # compare HF backends only
    python scripts/run_compare_summarisers.py --limit 25

    # include OpenAI (needs OPENAI_API_KEY)
    python scripts/run_compare_summarisers.py --limit 25 --openai

Outputs (into data/outputs/):
    summariser_comparison_<timestamp>.csv   — every (paper, model) row
    summariser_comparison_<timestamp>.json  — per-model aggregates

The JSON aggregates (mean latency, mean compression, mean ROUGE-L, total
estimated cost) are slide-ready: they are the evidence for the
"alternatives tried" part of the rubric.
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
from xuanzhi.nlp.compare import compare_summarisers, summarise_report, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare summariser backends.")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "xuanzhi.db")
    parser.add_argument(
        "--hf-models",
        nargs="+",
        default=["facebook/bart-large-cnn"],
        help="One or more HuggingFace summarisation model ids to compare.",
    )
    parser.add_argument("--openai", action="store_true", help="Include OpenAI.")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--limit", type=int, default=25, help="Papers to compare over.")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data" / "outputs", help="Report directory."
    )
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

    summarisers = [HFSummariser(model_name=m) for m in args.hf_models]
    if args.openai:
        summarisers.append(OpenAISummariser(model_name=args.openai_model))

    print(f"Comparing {len(summarisers)} summariser(s) over {len(papers)} papers...")
    df = compare_summarisers(summarisers, papers, max_words=args.max_words)

    if df.empty:
        print("No rows produced — every summariser errored. Check logs with --debug.")
        return

    csv_path, json_path = write_report(df, args.out)

    report = summarise_report(df)
    print("\n=== per-model summary ===")
    for model, stats in report["models"].items():
        print(
            f"  {model:26s}  "
            f"lat {stats['mean_latency_s']:6.2f}s  "
            f"words {stats['mean_word_count']:5.1f}  "
            f"compress {stats['mean_compression_ratio']:.2f}  "
            f"rougeL {stats['mean_rouge_l']}  "
            f"cost ${stats['total_est_cost_usd']:.4f}"
        )
    print(f"\nWrote:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
