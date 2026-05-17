# Setup

Everything you need to get Xuanzhi running on your machine. **Read this
before you `pip install` anything** — the dependency pins are tight on
purpose and the Intel-Mac caveats are real.

## Prerequisites

| Tool       | Version              | Why                                    |
|------------|----------------------|----------------------------------------|
| Python     | **3.12** (not 3.13)  | Every ML wheel has 3.12 coverage; 3.13 doesn't yet for some packages. |
| git        | any recent           | You're reading this in a repo.         |
| Disk space | ~3 GB free           | torch, CLIP, sentence-transformer, HF model caches add up. |

Check your Python with:

```bash
python3.12 --version    # should print 3.12.x
```

If you don't have 3.12: `brew install python@3.12` on macOS, or
`apt install python3.12 python3.12-venv` on Debian/Ubuntu.

## 1. Clone and create a venv

```bash
git clone https://github.com/devMYurge/Xuanzhi.git
cd Xuanzhi
python3.12 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your shell prompt.

## 2. Install dependencies

**Important**: upgrade pip first. Old pip is worse at finding prebuilt
wheels and falls back to source builds, which is where most install
pain comes from.

```bash
pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
```

The full install pulls ~2–3 GB (torch is the bulk). Give it a few
minutes the first time.

If `pip install` errors, jump to the [Troubleshooting](#troubleshooting)
section before re-running — most failures are about one bad package and
fix cleanly with a small adjustment.

## 3. Configure environment variables

The app reads API keys from a `.env` file at the repo root. **Do not
commit this file** — it's already in `.gitignore`.

Create `.env`:

```bash
cp .env.example .env   # then edit .env with your keys
```

If `.env.example` is missing, create `.env` yourself with this content:

```
# OpenAI — only needed if you use the --openai comparison or in-app summaries
OPENAI_API_KEY=sk-...

# Semantic Scholar — optional; raises your rate limit
# Free key: https://www.semanticscholar.org/product/api
SEMANTIC_SCHOLAR_API_KEY=
```

You **do not** strictly need either key to use the app — the HF
summariser, Playwright ingest, embeddings, clustering, and figure
pipeline all work without any keys. The keys only unlock:

- `--openai` flag on summarisation and the comparison harness.
- Higher Semantic Scholar API rate limits (avoids 429s on a busy demo).

## 4. Verify the install

This one-liner confirms the big-four ML packages are coherent:

```bash
python -c "import numpy, torch, transformers, openai, open_clip; \
print('numpy        ', numpy.__version__); \
print('torch        ', torch.__version__); \
print('transformers ', transformers.__version__); \
print('openai       ', openai.__version__); \
print('open_clip    ', open_clip.__version__); \
import torch.nn; print('torch+numpy   OK')"
```

You want to see roughly:

```
numpy         1.26.x
torch         2.2.x
transformers  4.43.x
openai        1.x
open_clip     2.x
torch+numpy   OK
```

If you see a "_compiled using NumPy 1.x cannot be run in NumPy 2.x_"
warning, your venv is in the broken state we hit during build. See
[Troubleshooting → NumPy 2.x mismatch](#numpy-2x--torch-2x-mismatch).

## 5. Run the pipeline once

The one-shot demo populates the SQLite DB end-to-end. Quick smoke first
(~5 minutes, ~10 papers, 5 figures, no API keys needed):

```bash
python scripts/run_demo.py --preset quick
```

You should see banners for each pipeline step (ingest → embed/cluster →
summarise → comparison → figures) and a final dashboard. If any step
prints `[FAIL]`, the dashboard tells you what went wrong but the rest
of the pipeline continues so you still get *some* demo data.

For a fuller corpus (~50 papers, with the OpenAI comparison):

```bash
python scripts/run_demo.py --preset full --openai
```

## 6. Launch the app

```bash
streamlit run streamlit_app.py
```

A browser tab opens at `http://localhost:8501`. Eight views in the sidebar:

- **Knowledge Graph** — the headline view. Interactive pyvis network;
  click a node for the in-graph info card.
- **Collection** — papers you've saved via the ☆ button.
- **Overview** — DB dashboard + pipeline status checklist.
- **Ingest** — Semantic Scholar quick-add (Playwright runs from the CLI).
- **Pipeline** — run every `scripts/` stage (ingest, embed + cluster,
  summarise, compare, extract figures) from the browser with a button
  per step; each step's CLI options are under "Advanced options".
- **Paper Explorer** — search, summaries, figures, related papers.
- **Cross-Literature** — two areas in, shared concepts + bridging papers.
- **Figure Source Lookup** — upload an image, get a formatted citation.

If a view says "no data yet" with a `python scripts/...` hint, run that
command in a terminal — or run the same step from the **Pipeline** view —
and reload the page.

---

## Troubleshooting

### `llvmlite` / `numba` build fails

You probably have an older `requirements.txt` or are reinstalling
something that pulls `umap-learn`. We removed `umap-learn` deliberately
because it drags in `numba` → `llvmlite`, which often won't build on
macOS. If you need a 2-D embedding-space plot, use `sklearn.manifold.TSNE`
(already installed). Never re-add `umap-learn`.

### `No matching distribution found for torch>=2.4`

You're on an **Intel Mac** (x86_64 CPU). PyTorch dropped Intel-Mac wheels
starting with torch 2.3 — 2.2.2 is the last version that installs on
x86_64 macOS. Our pinned `torch>=2.2,<2.3` is set up for exactly this.
If you see this error, double-check `requirements.txt` says `torch>=2.2,<2.3`
and that you ran `pip install --upgrade pip` first.

Are you actually on Apple Silicon hardware? Check with:

```bash
python -c "import platform; print(platform.machine())"
```

If it prints `arm64` you're on Apple Silicon and accidentally installed
the Intel Python (running under Rosetta). Reinstall Python natively:

```bash
arch -arm64 brew install python@3.12
rm -rf .venv
arch -arm64 python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

After this you can loosen the torch pin to `>=2.4,<2.6` locally for
better MPS performance, but don't push that change — it would break
Intel-Mac teammates.

### NumPy 2.x / torch 2.x mismatch

If verification prints "_compiled using NumPy 1.x cannot be run in
NumPy 2.x_" your venv has the wrong NumPy version. `pip install -r
requirements.txt` over an existing broken venv sometimes fails to
downgrade NumPy aggressively. Fix:

```bash
deactivate
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This is the universal "nuke and rebuild" fix when an environment
goes sideways. Takes ~5 minutes and resolves about 90% of dependency
weirdness in one shot.

### `streamlit-agraph` errors

We migrated the Knowledge Graph view from `streamlit-agraph` to `pyvis`.
The old package is still listed in `requirements.txt` (harmless) but
no code path uses it. If you get an import error mentioning agraph,
you're on an old branch — pull `main`.

### Semantic Scholar 429 rate-limit

The Ingest view hits Semantic Scholar's free anonymous tier, which is
aggressive about rate-limiting from shared IPs. Two fixes:

1. **Quick**: switch to the ArXiv CLI for that ingest run:
   ```bash
   python scripts/run_arxiv_ingest.py "your query" --max 30
   ```
2. **Better**: register for a free Semantic Scholar API key at
   <https://www.semanticscholar.org/product/api> and add it to your
   `.env` as `SEMANTIC_SCHOLAR_API_KEY=...`.

### Playwright "Executable doesn't exist" on first run

You skipped `playwright install chromium`. Run it now (no `pip`
required):

```bash
playwright install chromium
```

---

## Team conventions

- **Same Python everywhere.** Everyone uses Python 3.12 from the same
  venv recipe. If your verification one-liner prints different versions
  from teammates', stop and align before running the demo.
- **`.env` is never committed.** Keys live locally; share new variables
  via `.env.example`.
- **`data/` is gitignored.** Local SQLite DBs, downloaded PDFs, and
  extracted figures stay on each person's machine. Don't try to commit
  them.
- **Pin tightening is intentional.** If `pip` rejects a version, look
  at `requirements.txt` *before* loosening any bound — the comments next
  to each pin explain why.
