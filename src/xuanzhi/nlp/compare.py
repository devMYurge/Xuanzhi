"""Side-by-side comparison harness for summarisers.

The AIML rubric weights "general effort to the task, e.g. trying several
alternatives". This module makes that effort *measurable*: it runs every
summariser backend over the same set of papers and tabulates

* latency per paper (wall-clock seconds),
* output length (words / characters),
* compression ratio vs the source abstract,
* optional ROUGE-L against the source abstract (lexical overlap proxy),
* estimated cost (OpenAI only; HF is free/local).

The output is a tidy pandas DataFrame plus a JSON report dropped in
``data/outputs/`` so it can go straight into the slide deck.

This is descriptive, not a quality verdict — ROUGE against the abstract
rewards extractive overlap, which is not the same as a *good* summary.
The slides should pair these numbers with a couple of hand-picked
qualitative examples.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from xuanzhi.schema import Paper

from .summarisation import BaseSummariser, timed_summarise

log = logging.getLogger(__name__)

# Rough OpenAI gpt-4o-mini pricing used only for an order-of-magnitude
# cost column. Update if OpenAI pricing changes; HF rows are always 0.0.
_OPENAI_USD_PER_1K_INPUT = 0.00015
_OPENAI_USD_PER_1K_OUTPUT = 0.00060


@dataclass
class ComparisonRow:
    """One (paper, summariser) measurement."""

    paper_id: str
    title: str
    model: str
    summary: str
    latency_s: float
    word_count: int
    char_count: int
    compression_ratio: float  # summary_words / source_words
    rouge_l: float | None
    est_cost_usd: float


def _word_count(text: str) -> int:
    return len(text.split())


def _rouge_l(summary: str, reference: str) -> float | None:
    """ROUGE-L F1 via longest-common-subsequence. Returns None if either
    side is empty. Pure-stdlib so the harness has no extra dependency.
    """
    s = summary.lower().split()
    r = reference.lower().split()
    if not s or not r:
        return None
    # LCS length via DP.
    dp = [[0] * (len(r) + 1) for _ in range(len(s) + 1)]
    for i in range(1, len(s) + 1):
        for j in range(1, len(r) + 1):
            if s[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(s)][len(r)]
    if lcs == 0:
        return 0.0
    precision = lcs / len(s)
    recall = lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def _est_cost(model: str, source_text: str, summary: str) -> float:
    """Order-of-magnitude cost estimate. Non-OpenAI models are free."""
    # OpenAI chat models start with 'gpt-' (gpt-4o-mini, gpt-4o, gpt-5...).
    if not model.startswith("gpt"):
        return 0.0
    # ~1.3 tokens/word heuristic.
    in_tokens = _word_count(source_text) * 1.3
    out_tokens = _word_count(summary) * 1.3
    return (
        in_tokens / 1000 * _OPENAI_USD_PER_1K_INPUT
        + out_tokens / 1000 * _OPENAI_USD_PER_1K_OUTPUT
    )


def compare_summarisers(
    summarisers: list[BaseSummariser],
    papers: list[Paper],
    *,
    max_words: int = 80,
    compute_rouge: bool = True,
):
    """Run every summariser over every paper; return a pandas DataFrame.

    Errors from a single (paper, model) pair are logged and skipped so
    one bad row doesn't sink the whole run.
    """
    import pandas as pd

    rows: list[ComparisonRow] = []
    for paper in papers:
        source_text = paper.abstract or paper.title
        source_words = max(1, _word_count(source_text))
        for summariser in summarisers:
            try:
                summary, latency = timed_summarise(
                    summariser, source_text, max_words=max_words
                )
            except Exception as e:  # noqa: BLE001 — keep the harness resilient
                log.warning(
                    "[compare] %s failed on %s: %s",
                    summariser.model_name,
                    paper.id,
                    e,
                )
                continue
            wc = _word_count(summary)
            rows.append(
                ComparisonRow(
                    paper_id=paper.id,
                    title=paper.title,
                    model=summariser.model_name,
                    summary=summary,
                    latency_s=round(latency, 3),
                    word_count=wc,
                    char_count=len(summary),
                    compression_ratio=round(wc / source_words, 3),
                    rouge_l=(
                        round(_rouge_l(summary, source_text) or 0.0, 3)
                        if compute_rouge
                        else None
                    ),
                    est_cost_usd=round(_est_cost(summariser.model_name, source_text, summary), 6),
                )
            )
    return pd.DataFrame([asdict(r) for r in rows])


def summarise_report(df) -> dict:
    """Aggregate the per-row DataFrame into a per-model summary dict."""
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_papers": int(df["paper_id"].nunique()) if len(df) else 0,
        "models": {},
    }
    if len(df) == 0:
        return report
    for model, group in df.groupby("model"):
        report["models"][model] = {
            "n_summaries": int(len(group)),
            "mean_latency_s": round(float(group["latency_s"].mean()), 3),
            "mean_word_count": round(float(group["word_count"].mean()), 1),
            "mean_compression_ratio": round(float(group["compression_ratio"].mean()), 3),
            "mean_rouge_l": (
                round(float(group["rouge_l"].mean()), 3)
                if group["rouge_l"].notna().any()
                else None
            ),
            "total_est_cost_usd": round(float(group["est_cost_usd"].sum()), 6),
        }
    return report


def write_report(df, out_dir: Path) -> tuple[Path, Path]:
    """Persist the raw rows (CSV) and the aggregate (JSON). Returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"summariser_comparison_{stamp}.csv"
    json_path = out_dir / f"summariser_comparison_{stamp}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(summarise_report(df), indent=2), encoding="utf-8"
    )
    log.info("[compare] wrote %s and %s", csv_path.name, json_path.name)
    return csv_path, json_path
