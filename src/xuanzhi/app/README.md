# `xuanzhi.app`

The Streamlit UI — the rubric's required user interface, and the layer
where ingest + NLP + CV + graph all converge into something you can
click through.

## Run

```bash
streamlit run streamlit_app.py        # from the repo root
```

## Views

| View                  | What it shows | Needs |
|-----------------------|---------------|-------|
| Overview              | DB dashboard: paper/figure counts, sources, areas, pipeline status | nothing |
| Ingest                | Add papers live via the Semantic Scholar API | network |
| Knowledge Graph       | networkx graph of the corpus, filterable by edge type / year / size | papers |
| Paper Explorer        | Search papers; per-paper summaries, figures, concepts, related papers | papers (richer with NLP/CV) |
| Cross-Literature      | Pick two areas → shared concepts + bridging papers | papers + concepts |
| Figure Source Lookup  | Upload an image → ranked source papers to cite | extracted + indexed figures |

Every view checks its prerequisites and shows the exact command to run
if data is missing — so the app demos cleanly at any pipeline stage.

## Structure

- `main.py` — the app: sidebar router + the six view functions.
- `data.py` — cached data access (`st.cache_resource` for the Store and
  figure index, `st.cache_data` for query results; a session token
  busts the cache after a live ingest).
- `components.py` — shared renderers: paper card, summaries block,
  figure grid, the streamlit-agraph graph renderer (with a table
  fallback if the package is missing).

## Notes worth defending in the slides

- **Why Semantic Scholar (not Playwright) for in-app ingest:** Playwright
  drives a real browser and is best run as a CLI outside the Streamlit
  process. The REST API ingest is synchronous-friendly and safe to call
  from a button.
- **Why a table fallback in the graph view:** `streamlit-agraph` is the
  nice interactive renderer, but the app must not hard-crash if it is
  not installed — graceful degradation is a deliberate design choice.
