"""Topic / field classification of papers.

Why zero-shot
-------------
Source metadata gives us ArXiv categories and Semantic Scholar fields,
but they are (a) coarse, (b) inconsistent across sources, and (c) absent
for some papers. A zero-shot classifier lets us project every paper onto
*our own* controlled label set without training data — exactly the
situation a 4-day prototype is in.

Model: ``facebook/bart-large-mnli`` runs natural-language-inference
style zero-shot classification. We pose each candidate label as a
hypothesis ("This paper is about {label}") and read off the entailment
probability.

Swap-in path: once we have a labelled slice, a fine-tuned
``allenai/scibert_scivocab_uncased`` sequence classifier drops in behind
the same ``classify`` interface. That fine-tune is the natural
"alternative we tried" story for the slides.
"""

from __future__ import annotations

import logging

from xuanzhi.schema import Paper, ResearchArea

from ._device import resolve_device

log = logging.getLogger(__name__)

# A reasonable default label set for a CS/ML-leaning demo corpus. Override
# per-run with whatever taxonomy the group's chosen corpus needs.
DEFAULT_LABELS = [
    "machine learning",
    "natural language processing",
    "computer vision",
    "information retrieval",
    "knowledge graphs",
    "human-computer interaction",
    "robotics",
    "computational biology",
    "theory of computation",
    "systems and networking",
]


class ZeroShotClassifier:
    """HuggingFace zero-shot classification wrapper.

    Parameters
    ----------
    model_name:
        Any NLI model compatible with the ``zero-shot-classification``
        pipeline. Default ``facebook/bart-large-mnli``.
    device:
        ``"cuda" | "mps" | "cpu" | None`` (auto).
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = resolve_device(device)
        self._pipe = None  # lazy

    def _ensure_loaded(self):
        if self._pipe is None:
            from transformers import pipeline

            # transformers wants an int device index for cuda, -1 for cpu;
            # "mps" is accepted as a string in recent versions.
            if self.device == "cuda":
                device_arg: object = 0
            elif self.device == "mps":
                device_arg = "mps"
            else:
                device_arg = -1

            log.info("[nlp] loading zero-shot %s on %s", self.model_name, self.device)
            self._pipe = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=device_arg,
            )
        return self._pipe

    # ------------------------------------------------------------ classify

    def classify(
        self,
        text: str,
        candidate_labels: list[str],
        multi_label: bool = True,
    ) -> dict[str, float]:
        """Return ``{label: score}`` sorted high → low."""
        pipe = self._ensure_loaded()
        out = pipe(text, candidate_labels, multi_label=multi_label)
        return dict(zip(out["labels"], out["scores"]))

    def classify_paper(
        self,
        paper: Paper,
        candidate_labels: list[str] | None = None,
        threshold: float = 0.5,
        top_k: int = 3,
    ) -> list[tuple[ResearchArea, float]]:
        """Classify a paper into ResearchArea objects.

        Returns ``[(ResearchArea, confidence)]`` for every label above
        ``threshold`` (capped at ``top_k``). The areas carry
        ``source="zero-shot"`` so they're distinguishable from
        source-supplied categories in the DB.
        """
        labels = candidate_labels or DEFAULT_LABELS
        text = f"{paper.title}. {paper.abstract or ''}".strip()
        scores = self.classify(text, labels, multi_label=True)
        results: list[tuple[ResearchArea, float]] = []
        for label, score in list(scores.items())[:top_k]:
            if score >= threshold:
                area = ResearchArea.make(label, source="zero-shot")
                results.append((area, float(score)))
        return results
