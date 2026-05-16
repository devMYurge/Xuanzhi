"""Sentence embeddings for papers.

Why sentence-transformers
-------------------------
``sentence-transformers`` packages MiniLM / MPNet / SPECTER2 (the
scientific-text model) behind a uniform API and ships with
mean-pooling + L2 normalisation baked in. For abstract-length text on a
laptop, ``all-MiniLM-L6-v2`` (384-d, ~25MB) is the right default; for
scientific writing specifically, ``allenai/specter2_base`` is a stronger
swap-in. Both work with the same code path here.

How embeddings flow
-------------------
1. :class:`Embedder` is a stateless wrapper over the chosen model.
2. ``encode_papers`` produces a numpy matrix, one row per paper.
3. :func:`embed_corpus` persists each row into
   ``papers_embeddings(paper_id, model, vector_bytes)`` so downstream
   clustering / similarity / Streamlit code never re-encodes.
4. :class:`EmbeddingMatrix` is the loader on the other side — pulls
   everything for a given model back into a dense numpy matrix plus the
   paper-id ordering, ready for sklearn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from xuanzhi.db import Store
from xuanzhi.schema import Paper

from ._device import resolve_device

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- Embedder


class Embedder:
    """Thin wrapper around a sentence-transformers model.

    Parameters
    ----------
    model_name:
        Any HF model id supported by sentence-transformers. Default
        ``"sentence-transformers/all-MiniLM-L6-v2"`` is small + fast.
        Try ``"allenai/specter2_base"`` for sharper scientific
        similarity (slower, larger).
    device:
        ``"cuda" | "mps" | "cpu" | None``. ``None`` auto-detects.
    batch_size:
        Forwarded to ``model.encode``.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = None  # lazy

    # ------------------------------------------------ lazy model handle

    def _ensure_loaded(self):
        if self._model is None:
            # Imported lazily so importing xuanzhi.nlp.embeddings doesn't
            # pull in transformers/torch.
            from sentence_transformers import SentenceTransformer

            log.info("[nlp] loading %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    # ---------------------------------------------------------- encode

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(N, D)`` float32 array of L2-normalised vectors."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._ensure_loaded()
        vecs = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32, copy=False)

    def encode_papers(self, papers: Sequence[Paper]) -> np.ndarray:
        """Concatenate title + abstract per paper, then encode."""
        texts = [_paper_text(p) for p in papers]
        return self.encode(texts)

    @property
    def dim(self) -> int:
        return int(self._ensure_loaded().get_sentence_embedding_dimension())


def _paper_text(paper: Paper) -> str:
    """Canonical 'embed-this' projection of a paper.

    Title carries strong topical signal; abstract carries the detail.
    Joining them with a sentinel keeps them separable for downstream
    inspection without much effect on the embedding itself.
    """
    abstract = paper.abstract or ""
    return f"{paper.title}. {abstract}".strip()


# ----------------------------------------------------------- corpus pass


def embed_corpus(
    store: Store,
    embedder: Embedder,
    *,
    only_missing: bool = True,
    batch_size: int = 32,
) -> int:
    """Embed every paper in the DB (optionally only the ones not yet
    embedded under ``embedder.model_name``) and persist the vectors.

    Returns the number of papers freshly embedded.
    """
    if only_missing:
        ids = set(store.paper_ids_missing_embedding(embedder.model_name))
    else:
        ids = None

    batch_papers: list[Paper] = []
    n_done = 0
    for paper in store.iter_papers():
        if ids is not None and paper.id not in ids:
            continue
        batch_papers.append(paper)
        if len(batch_papers) >= batch_size:
            n_done += _flush(store, embedder, batch_papers)
            batch_papers = []
    if batch_papers:
        n_done += _flush(store, embedder, batch_papers)
    return n_done


def _flush(store: Store, embedder: Embedder, batch: list[Paper]) -> int:
    vecs = embedder.encode_papers(batch)
    dim = int(vecs.shape[1])
    for paper, vec in zip(batch, vecs):
        store.put_embedding(
            paper_id=paper.id,
            model=embedder.model_name,
            vector=vec.astype(np.float32, copy=False).tobytes(),
            dim=dim,
        )
    log.info("[nlp] embedded %d papers", len(batch))
    return len(batch)


# ----------------------------------------------------------- loader side


@dataclass
class EmbeddingMatrix:
    """Dense matrix of stored embeddings, aligned with a ``paper_ids`` list.

    Returned by :meth:`from_store` so downstream sklearn code (clustering,
    nearest-neighbours, etc.) can stay completely DB-agnostic.
    """

    paper_ids: list[str]
    vectors: np.ndarray
    model: str

    def __len__(self) -> int:
        return len(self.paper_ids)

    @classmethod
    def from_store(cls, store: Store, model: str) -> "EmbeddingMatrix":
        ids: list[str] = []
        rows: list[np.ndarray] = []
        for paper_id, dim, blob in store.iter_embeddings(model):
            ids.append(paper_id)
            rows.append(np.frombuffer(blob, dtype=np.float32).reshape(dim))
        vectors = np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        return cls(paper_ids=ids, vectors=vectors, model=model)

    def cosine_similar(self, query_idx: int, top_k: int = 10) -> list[tuple[str, float]]:
        """Return ``[(paper_id, similarity)]`` for the top_k nearest neighbours
        of ``self.vectors[query_idx]``. Vectors are already L2-normalised
        by the Embedder, so dot product is cosine similarity.
        """
        if len(self) == 0:
            return []
        sims = self.vectors @ self.vectors[query_idx]
        order = np.argsort(-sims)
        results: list[tuple[str, float]] = []
        for i in order:
            if int(i) == query_idx:
                continue
            results.append((self.paper_ids[int(i)], float(sims[int(i)])))
            if len(results) >= top_k:
                break
        return results
