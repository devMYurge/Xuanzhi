"""Unsupervised clustering of papers + concept derivation.

This is where scikit-learn does the heavy lifting. The pipeline:

1. Take the stored embedding matrix (:class:`xuanzhi.nlp.EmbeddingMatrix`).
2. Cluster it — KMeans (fast, needs ``k``) or HDBSCAN (density-based,
   finds ``k`` itself and flags noise as ``-1``).
3. Name each cluster from the **TF-IDF top terms** of its papers'
   titles + abstracts — a cheap, interpretable label without an LLM.
4. Turn each named cluster into a :class:`xuanzhi.schema.Concept` and
   attach papers to it via PaperConcept edges.

The result is the backbone of the cross-literature graph: papers in
different ArXiv categories that land in the same embedding cluster are
exactly the "bridging" connections the product is meant to surface.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

import numpy as np

from xuanzhi.db import Store
from xuanzhi.schema import Concept

from .embeddings import EmbeddingMatrix

log = logging.getLogger(__name__)

ClusterMethod = Literal["kmeans", "hdbscan"]


# ----------------------------------------------------------------- cluster


def cluster_embeddings(
    matrix: EmbeddingMatrix,
    method: ClusterMethod = "kmeans",
    *,
    k: int | None = None,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster an embedding matrix and return an integer label per row.

    Parameters
    ----------
    method:
        ``"kmeans"`` — needs ``k``; if ``k`` is None we pick a heuristic
        ``k = round(sqrt(n / 2))``.
        ``"hdbscan"`` — density-based; finds the cluster count itself and
        labels outliers ``-1``.
    k:
        Cluster count for KMeans. Ignored by HDBSCAN.
    min_cluster_size:
        HDBSCAN's minimum cluster size.

    Returns
    -------
    np.ndarray of shape ``(len(matrix),)`` with integer cluster labels.
    """
    n = len(matrix)
    if n == 0:
        return np.zeros((0,), dtype=int)

    if method == "kmeans":
        from sklearn.cluster import KMeans

        if k is None:
            k = max(2, round((n / 2) ** 0.5))
        k = min(k, n)
        log.info("[nlp] KMeans k=%d over %d papers", k, n)
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        return model.fit_predict(matrix.vectors)

    if method == "hdbscan":
        try:
            import hdbscan
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "hdbscan is not installed — `pip install hdbscan` or use method='kmeans'."
            ) from e

        log.info("[nlp] HDBSCAN min_cluster_size=%d over %d papers", min_cluster_size, n)
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",  # vectors are L2-normalised, so this ~ cosine
        )
        return model.fit_predict(matrix.vectors)

    raise ValueError(f"unknown clustering method: {method!r}")


# ----------------------------------------------------------- concept names


_TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was",
    "our", "use", "using", "used", "can", "based", "via", "new", "paper",
    "results", "approach", "method", "methods", "model", "models", "show",
    "propose", "proposed", "present", "study", "data", "task", "tasks",
}


def _top_terms_per_cluster(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 4,
) -> dict[int, str]:
    """TF-IDF top terms per cluster → a human-readable concept name.

    Uses sklearn's TfidfVectorizer fitted on the whole corpus, then
    averages tf-idf weight within each cluster and takes the top terms.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectoriser = TfidfVectorizer(
        lowercase=True,
        token_pattern=_TOKEN_RE.pattern,
        stop_words=list(_STOPWORDS),
        max_df=0.6,
        min_df=2,
        ngram_range=(1, 2),
    )
    try:
        tfidf = vectoriser.fit_transform(texts)
    except ValueError:
        # Corpus too small / vocabulary empty — fall back to numeric names.
        return {int(c): f"cluster {int(c)}" for c in set(labels.tolist())}

    vocab = np.array(vectoriser.get_feature_names_out())
    names: dict[int, str] = {}
    for cluster in sorted(set(labels.tolist())):
        if cluster == -1:
            names[cluster] = "unclustered"
            continue
        rows = np.where(labels == cluster)[0]
        if len(rows) == 0:
            names[cluster] = f"cluster {cluster}"
            continue
        mean_weights = np.asarray(tfidf[rows].mean(axis=0)).ravel()
        top_idx = np.argsort(-mean_weights)[:top_n]
        terms = [vocab[i] for i in top_idx if mean_weights[i] > 0]
        names[cluster] = ", ".join(terms) if terms else f"cluster {cluster}"
    return names


# -------------------------------------------------------- derive concepts


def derive_concepts_from_clusters(
    store: Store,
    matrix: EmbeddingMatrix,
    labels: np.ndarray,
    *,
    persist: bool = True,
) -> dict[int, Concept]:
    """Name each cluster and (optionally) write Concept + PaperConcept rows.

    Returns ``{cluster_label: Concept}`` (excluding the ``-1`` noise
    cluster). When ``persist`` is True, every paper is linked to its
    cluster's concept in the DB.
    """
    # Pull title+abstract text for each paper, in matrix order.
    texts: list[str] = []
    for pid in matrix.paper_ids:
        paper = store.get_paper(pid)
        if paper is None:
            texts.append("")
        else:
            texts.append(f"{paper.title}. {paper.abstract or ''}")

    names = _top_terms_per_cluster(texts, labels)

    concepts: dict[int, Concept] = {}
    for cluster, name in names.items():
        if cluster == -1:
            continue
        concepts[cluster] = Concept.make(name, extraction_source="cluster")

    if persist:
        for pid, label in zip(matrix.paper_ids, labels.tolist()):
            concept = concepts.get(int(label))
            if concept is not None:
                store.add_concept(pid, concept, salience=1.0)
        log.info("[nlp] persisted %d cluster-concepts", len(concepts))

    return concepts
