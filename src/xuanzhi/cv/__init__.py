"""Computer-vision layer: figure extraction, classification, and the
image-to-source lookup.

The flagship feature lives here. A researcher reusing a figure can upload
the image and get the source paper back for proper citation — this is
the most novel thing Xuanzhi does, and it is pure CV:

    PDF  --extract-->  Figure images  --CLIP-->  embedding index
                                                       |
    uploaded image  --CLIP-->  query vector  --kNN-----+--> source paper

Modules
-------
    pdf_download.py  Fetch a paper's PDF (httpx, polite, cached on disk).
    figures.py       PyMuPDF figure extraction: images + bbox + caption.
    classify.py      CLIP zero-shot figure-type tagging.
    index.py         CLIP image embeddings + sklearn kNN + lookup.

All of this uses HuggingFace vision models (CLIP via sentence-transformers)
and scikit-learn — the course libraries. A torchvision-trained classifier
is the documented alternative path for classify.py.
"""

from .classify import FIGURE_TYPE_PROMPTS, FigureClassifier
from .figures import extract_figures
from .index import FigureIndex, FigureMatch
from .pdf_download import download_pdf

__all__ = [
    "FIGURE_TYPE_PROMPTS",
    "FigureClassifier",
    "FigureIndex",
    "FigureMatch",
    "download_pdf",
    "extract_figures",
]
