# `xuanzhi.nlp`

The NLP layer. Everything here uses **HuggingFace transformers** and
**scikit-learn** — the course libraries the AIML rubric expects to see
actively used — with frontier-model (OpenAI) paths alongside the HF ones
so comparisons are head-to-head.

## Modules

| Module             | Course library            | What it does |
|--------------------|---------------------------|--------------|
| `embeddings.py`    | sentence-transformers     | Encode title+abstract → vectors; persist to `paper_embeddings`; load back as a matrix for sklearn. |
| `classification.py`| transformers (zero-shot)  | `bart-large-mnli` projects papers onto a controlled label set → `ResearchArea` rows with confidence. |
| `summarisation.py` | transformers + openai     | `BaseSummariser` interface; `HFSummariser` (`bart-large-cnn`) and `OpenAISummariser` (`gpt-4o-mini`) both write `Summary` rows. |
| `clustering.py`    | scikit-learn              | KMeans / HDBSCAN over embeddings; TF-IDF top-terms name each cluster → `Concept` + `PaperConcept`. |
| `compare.py`       | pandas (+ stdlib ROUGE)   | Runs every summariser over the same papers; tabulates latency / length / compression / ROUGE-L / cost. |

## Design choices worth defending in the slides

- **Why zero-shot classification:** source categories (ArXiv, S2) are
  coarse, inconsistent, and sometimes missing. Zero-shot lets us project
  every paper onto *our* taxonomy with no training data — the right call
  for a days-long prototype. Fine-tuned SciBERT is the documented
  "next alternative".
- **Why cluster → concept:** embedding clusters that span different
  ArXiv categories *are* the cross-literature bridges the product is
  built to surface. TF-IDF naming keeps the labels interpretable
  without an LLM in the loop.
- **Why a comparison harness:** the rubric rewards "trying several
  alternatives". `compare.py` turns that into numbers — HF is free /
  offline / slower, OpenAI gpt-4o-mini is fluent / costs a fraction of
  a cent per abstract / needs network. The JSON report is slide-ready.

## Run order

```bash
python scripts/run_arxiv_ingest.py "graph rag" --max 50   # populate DB
python scripts/run_embed_papers.py --cluster kmeans       # embed + cluster
python scripts/run_summarise.py --limit 20                # HF summaries
python scripts/run_compare_summarisers.py --limit 25 --openai  # comparison report
```
