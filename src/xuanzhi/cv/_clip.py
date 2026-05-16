"""Shared CLIP handle for the CV layer.

CLIP encodes images *and* text into the same vector space. That single
property powers both jobs in this package:

* :mod:`xuanzhi.cv.classify` — embed an image, embed candidate type
  prompts, compare → zero-shot figure-type classification.
* :mod:`xuanzhi.cv.index` — embed every extracted figure, build a kNN
  index, embed an uploaded query image → nearest figures → source paper.

We load CLIP through ``sentence-transformers`` (``clip-ViT-B-32``) so the
CV layer reuses the exact dependency the NLP layer already pulls in, and
the encode API is identical for images and text.

The model is loaded once and memoised per (model_name, device).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from xuanzhi.utils import resolve_device

log = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "clip-ViT-B-32"


@lru_cache(maxsize=4)
def get_clip(model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None):
    """Return a memoised sentence-transformers CLIP model.

    Imported lazily so importing the cv package doesn't drag in torch.
    """
    from sentence_transformers import SentenceTransformer

    dev = resolve_device(device)
    log.info("[cv] loading CLIP %s on %s", model_name, dev)
    return SentenceTransformer(model_name, device=dev)
