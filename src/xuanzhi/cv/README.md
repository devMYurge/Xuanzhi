# `xuanzhi.cv`

The computer-vision layer — and the home of Xuanzhi's flagship feature:
**image-to-source lookup**. Upload a figure you want to reuse, get the
source paper back for proper citation.

## The pipeline

```
PDF --download--> data/pdfs/{paper_id}.pdf
    --extract--> data/figures/{paper_id}/*.png  + Figure rows
    --classify--> figure_type (CLIP zero-shot)
    --embed-->   figure_embeddings (CLIP image vectors)

uploaded image --CLIP embed--> query vector --sklearn kNN--> source paper
```

## Modules

| Module           | Course library              | What it does |
|------------------|-----------------------------|--------------|
| `pdf_download.py`| httpx                       | Fetch + cache a paper's PDF on disk. |
| `figures.py`     | PyMuPDF                     | Extract embedded images + bbox + nearest caption; filter logos/icons. |
| `classify.py`    | HF vision (CLIP)            | Zero-shot figure-type tagging (chart / diagram / photo / table / equation). |
| `index.py`       | HF vision (CLIP) + sklearn  | CLIP image embeddings + `NearestNeighbors` index + image-to-source search. |

## Why these choices (defend in the slides)

- **CLIP for both jobs.** CLIP embeds images and text into one space, so
  the *same* model does zero-shot classification (image vs. type-prompt
  similarity) and the similarity index (image vs. image). One dependency,
  two features. Loaded via `sentence-transformers` so it reuses the NLP
  layer's dependency.
- **sklearn `NearestNeighbors`, not FAISS.** The demo corpus is hundreds
  to low-thousands of figures — brute-force cosine kNN is instant and
  needs no extra dependency. FAISS is the documented scale-up path.
- **Raster-image extraction only.** Vector figures, multi-panel layouts
  and text-rendered figures are *not* handled — a known, documented
  limitation. Modern arXiv PDFs embed most charts/photos as raster
  images, so coverage is good enough for a prototype demo.
- **Documented alternative.** A supervised `torchvision` classifier
  (fine-tuned ResNet on labelled figures) swaps in behind
  `FigureClassifier.classify` — the "alternative we tried" story.

## Run order

```bash
python scripts/run_arxiv_ingest.py "graph rag" --max 50   # papers (with pdf_url)
python scripts/run_extract_figures.py --limit 20          # download/extract/classify/index
python scripts/run_figure_lookup.py my_figure.png         # the demo
```
