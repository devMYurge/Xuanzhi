# Xuanzhi （玄之）

A cognition layer for academic research. Scrapes papers across multiple
sources, extracts structured knowledge (entities, claims, methods,
figures), stores it in a database, and surfaces it as a navigable
knowledge graph for cross-literature exploration.

AIML Analytics Group Project [30%] — IE University, PPLEDBA. Deadline
**17 May 2026**. Group: A. Rich, S. Klonis, S. Soliveres, M. Yu.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Then run the pipeline end to end:

```bash
# 1. ingest papers (Playwright scrape of ArXiv)
python scripts/run_arxiv_ingest.py "graph rag knowledge graph" --max 50

# 2. embed + cluster into concepts (sentence-transformers + sklearn)
python scripts/run_embed_papers.py --cluster kmeans

# 3. summarise — HF model, add --claude to compare against a frontier model
python scripts/run_summarise.py --limit 30
python scripts/run_compare_summarisers.py --limit 25 --claude

# 4. extract + classify + index figures (PyMuPDF + CLIP)
python scripts/run_extract_figures.py --limit 20
python scripts/run_figure_lookup.py some_figure.png

# 5. launch the UI
streamlit run streamlit_app.py
```

Everything writes to `data/xuanzhi.db` (SQLite). Inspect it with any
SQLite client, or programmatically:

```python
from xuanzhi.db import Store
store = Store("data/xuanzhi.db")
for p in store.iter_papers(limit=5):
    print(p.title, [a.name for a in p.authors])
```

## Layout

```
docs/
  academic/         AIML deliverables (report, slides, demo script, refs)
  planning/         Build brief, roadmap, decision records
  product/          Scaffold specs (cognitive primitives)
  research/         Background notes (GraphRAG, prior art)

src/xuanzhi/
  schema/           Unified Pydantic data model (Paper, Author, ...)
  ingest/           Playwright + REST ingest modules (ArXiv, S2, ...)
  db/               SQLite store + DDL
  nlp/              HuggingFace classification / summarisation + sklearn analytics
  cv/               PDF figure extraction + vision-model classification + source-lookup
  graph/            networkx graph construction & cross-literature queries
  app/              Streamlit UI
  utils/            Shared helpers (device detection)

streamlit_app.py    Streamlit entry point — `streamlit run streamlit_app.py`
data/               Local ingest corpus + SQLite DB (gitignored)
notebooks/          Exploratory analysis
scripts/            CLIs (run_arxiv_ingest.py, run_embed_papers.py, ...)
tests/              unit / integration / e2e
```

## Pillars and course-library mapping

| Pillar              | Course library                          | Module               |
|---------------------|-----------------------------------------|----------------------|
| Web scraping        | Playwright                              | `xuanzhi.ingest`     |
| Classification      | HuggingFace transformers                | `xuanzhi.nlp`        |
| Summarisation       | HuggingFace transformers + Claude       | `xuanzhi.nlp`        |
| Embeddings/cluster  | sentence-transformers + scikit-learn    | `xuanzhi.nlp`        |
| Figure CV           | torchvision / HF vision (CLIP/ViT)      | `xuanzhi.cv`         |
| UI                  | Streamlit                               | `xuanzhi.app`        |

See `docs/planning/project_brief.md` for the full build brief.
