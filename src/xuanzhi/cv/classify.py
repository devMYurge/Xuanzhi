"""Zero-shot figure-type classification with CLIP.

Same trick as the NLP zero-shot classifier, in the image domain: we
embed the figure image and a set of natural-language prompts ("a bar
chart or line graph", "a flow diagram", ...) into CLIP's shared space,
then assign the figure the type whose prompt is most similar.

No training data, no labelled figures — which is the right call for a
4-day prototype. The documented alternative is a supervised
``torchvision`` classifier (e.g. a fine-tuned ResNet) trained on a
labelled figure set; that swaps in behind the same ``classify`` method.
"""

from __future__ import annotations

import logging

from xuanzhi.schema import Figure, FigureType

from ._clip import DEFAULT_CLIP_MODEL, get_clip

log = logging.getLogger(__name__)

# Natural-language prompts, one per FigureType. CLIP matches the image
# against these; the best match wins. Prompts are intentionally verbose —
# CLIP responds better to descriptive phrases than to bare labels.
FIGURE_TYPE_PROMPTS: dict[FigureType, str] = {
    FigureType.CHART: "a data chart, bar chart, line graph, or scatter plot",
    FigureType.DIAGRAM: "a schematic diagram, flowchart, or model architecture figure",
    FigureType.PHOTO: "a photograph or natural image",
    FigureType.TABLE: "a table of numbers or text laid out in a grid",
    FigureType.EQUATION: "a mathematical equation or formula",
}


class FigureClassifier:
    """CLIP zero-shot classifier for figure types."""

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self._prompt_vectors = None  # lazy
        self._types: list[FigureType] = list(FIGURE_TYPE_PROMPTS.keys())

    def _ensure_prompts(self):
        if self._prompt_vectors is None:
            model = get_clip(self.model_name, self.device)
            prompts = [FIGURE_TYPE_PROMPTS[t] for t in self._types]
            self._prompt_vectors = model.encode(
                prompts, convert_to_numpy=True, normalize_embeddings=True
            )
        return self._prompt_vectors

    # ------------------------------------------------------------ classify

    def classify_image(self, image_path: str) -> tuple[FigureType, float]:
        """Return ``(FigureType, confidence)`` for one image file."""
        from PIL import Image

        model = get_clip(self.model_name, self.device)
        prompt_vecs = self._ensure_prompts()

        with Image.open(image_path) as im:
            img_vec = model.encode(
                [im.convert("RGB")],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]

        # cosine similarity (everything is L2-normalised)
        sims = prompt_vecs @ img_vec
        best = int(sims.argmax())
        return self._types[best], float(sims[best])

    def classify_figure(self, figure: Figure) -> Figure:
        """Return a copy of ``figure`` with ``figure_type`` filled in.

        On any read/decode error the figure is returned unchanged
        (``UNKNOWN``) so a single corrupt image doesn't sink a batch.
        """
        try:
            fig_type, _score = self.classify_image(figure.image_path)
        except Exception as e:  # noqa: BLE001
            log.warning("[cv] could not classify %s: %s", figure.image_path, e)
            return figure
        return figure.model_copy(update={"figure_type": fig_type})
