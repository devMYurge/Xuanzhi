"""Shared CLIP handle for the CV layer.

CLIP encodes images *and* text into the same vector space. That single
property powers both jobs in this package:

* :mod:`xuanzhi.cv.classify` — embed an image, embed candidate type
  prompts, compare → zero-shot figure-type classification.
* :mod:`xuanzhi.cv.index` — embed every extracted figure, build a kNN
  index, embed an uploaded query image → nearest figures → source paper.

We load CLIP through ``open_clip_torch`` (ViT-B/32, OpenAI weights) so
weights download from OpenAI's CDN rather than HuggingFace's Xet Storage.
The adapter exposes the same .encode() signature used by sentence-transformers
so the rest of the CV layer needs no changes.

The model is loaded once and memoised per (model_name, device).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from xuanzhi.utils import resolve_device

log = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "ViT-B-32"


class _OpenCLIPAdapter:
    """Wraps open_clip with a sentence-transformers-compatible .encode() API."""

    def __init__(self, arch: str, pretrained: str, device: str) -> None:
        import open_clip

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.model.eval()

    def encode(
        self,
        inputs: list,
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        import torch
        from PIL import Image as PILImage

        with torch.no_grad():
            if inputs and isinstance(inputs[0], str):
                tokens = self.tokenizer(inputs).to(self.device)
                feats = self.model.encode_text(tokens).float()
            else:
                tensors = torch.stack(
                    [
                        self.preprocess(
                            img if isinstance(img, PILImage.Image) else PILImage.open(img).convert("RGB")
                        )
                        for img in inputs
                    ]
                ).to(self.device)
                feats = self.model.encode_image(tensors).float()

            if normalize_embeddings:
                feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy() if convert_to_numpy else feats


@lru_cache(maxsize=4)
def get_clip(model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None) -> _OpenCLIPAdapter:
    """Return a memoised OpenCLIP adapter.

    Imported lazily so importing the cv package doesn't drag in torch.
    """
    dev = resolve_device(device)
    log.info("[cv] loading CLIP %s on %s", model_name, dev)
    return _OpenCLIPAdapter(arch=model_name, pretrained="openai", device=dev)
