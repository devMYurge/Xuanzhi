"""Figure embedding index + image-to-source lookup.

This is the flagship feature. The flow:

1. :meth:`FigureIndex.build_from_store` embeds every extracted figure
   with CLIP and persists the vectors into ``figure_embeddings``.
2. :meth:`FigureIndex.load` pulls those vectors back into a dense matrix
   and fits a scikit-learn ``NearestNeighbors`` index over them.
3. :meth:`FigureIndex.search` takes an *uploaded* image, embeds it with
   the same CLIP model, and returns the most similar stored figures —
   each carrying its source ``Paper`` so the user gets a citation.

Why sklearn NearestNeighbors (not FAISS)
----------------------------------------
The demo corpus is hundreds to low-thousands of figures. Brute-force
cosine kNN over that is instant, needs no extra dependency, and keeps
the course-library footprint (sklearn) front and centre. FAISS is the
documented scale-up path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from xuanzhi.db import Store
from xuanzhi.schema import Figure, Paper

from ._clip import DEFAULT_CLIP_MODEL, get_clip

log = logging.getLogger(__name__)


@dataclass
class FigureMatch:
    """One hit from an image-to-source lookup."""

    figure: Figure
    paper: Paper | None
    similarity: float  # cosine similarity in [-1, 1], higher = closer

    def citation_line(self) -> str:
        """A rough citation string for the matched source paper."""
        if self.paper is None:
            return "(source paper not found in database)"
        authors = ", ".join(a.name for a in self.paper.authors[:3])
        if len(self.paper.authors) > 3:
            authors += " et al."
        year = f" ({self.paper.year})" if self.paper.year else ""
        # Gracefully handle papers ingested without author/year metadata.
        prefix = f"{authors}{year}. " if authors or year else ""
        return f"{prefix}{self.paper.title}."


class FigureIndex:
    """CLIP-embedding kNN index over extracted figures."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._figure_ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._nn = None  # sklearn NearestNeighbors, fitted lazily

    # ------------------------------------------------------------- build

    def build_from_store(
        self,
        store: Store,
        *,
        only_missing: bool = True,
        batch_size: int = 16,
    ) -> int:
        """Embed figures with CLIP and persist vectors to the DB.

        Returns the number of figures freshly embedded.
        """
        from PIL import Image

        model = get_clip(self.model_name, self.device)

        if only_missing:
            wanted = set(store.figure_ids_missing_embedding(self.model_name))
        else:
            wanted = None

        batch: list[Figure] = []
        n_done = 0

        def _flush(items: list[Figure]) -> int:
            images, ok_items = [], []
            for fig in items:
                try:
                    with Image.open(fig.image_path) as im:
                        images.append(im.convert("RGB").copy())
                    ok_items.append(fig)
                except Exception as e:  # noqa: BLE001
                    log.warning("[cv] skipping unreadable %s: %s", fig.image_path, e)
            if not images:
                return 0
            vecs = model.encode(
                images, convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)
            dim = int(vecs.shape[1])
            for fig, vec in zip(ok_items, vecs):
                store.put_figure_embedding(
                    fig.id, self.model_name, vec.tobytes(), dim
                )
            return len(ok_items)

        for fig in store.iter_figures():
            if wanted is not None and fig.id not in wanted:
                continue
            batch.append(fig)
            if len(batch) >= batch_size:
                n_done += _flush(batch)
                batch = []
        if batch:
            n_done += _flush(batch)

        log.info("[cv] embedded %d figures", n_done)
        return n_done

    # -------------------------------------------------------------- load

    def load(self, store: Store) -> "FigureIndex":
        """Load all stored figure embeddings and fit the kNN index."""
        from sklearn.neighbors import NearestNeighbors

        ids: list[str] = []
        rows: list[np.ndarray] = []
        for figure_id, dim, blob in store.iter_figure_embeddings(self.model_name):
            ids.append(figure_id)
            rows.append(np.frombuffer(blob, dtype=np.float32).reshape(dim))

        self._figure_ids = ids
        self._vectors = (
            np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        )
        if len(ids):
            # cosine metric; vectors are already normalised so this is
            # equivalent to dot-product ranking.
            self._nn = NearestNeighbors(metric="cosine")
            self._nn.fit(self._vectors)
        log.info("[cv] figure index loaded: %d figures", len(ids))
        return self

    # ------------------------------------------------------------ search

    def search(
        self,
        store: Store,
        image_path: str,
        top_k: int = 5,
    ) -> list[FigureMatch]:
        """Find the stored figures most similar to an uploaded image.

        Returns ``FigureMatch`` objects (figure + source paper +
        similarity), best match first. This is the image-to-source
        citation lookup.
        """
        if self._nn is None or self._vectors is None or len(self._figure_ids) == 0:
            log.warning("[cv] search called on an empty index — call .load() first")
            return []

        from PIL import Image

        model = get_clip(self.model_name, self.device)
        with Image.open(image_path) as im:
            query = model.encode(
                [im.convert("RGB")],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

        k = min(top_k, len(self._figure_ids))
        distances, indices = self._nn.kneighbors(query, n_neighbors=k)

        matches: list[FigureMatch] = []
        for dist, idx in zip(distances[0], indices[0]):
            figure_id = self._figure_ids[int(idx)]
            figure = store.get_figure(figure_id)
            if figure is None:
                continue
            paper = store.get_paper(figure.paper_id)
            matches.append(
                FigureMatch(
                    figure=figure,
                    paper=paper,
                    similarity=float(1.0 - dist),  # cosine distance -> similarity
                )
            )
        return matches

    def __len__(self) -> int:
        return len(self._figure_ids)
