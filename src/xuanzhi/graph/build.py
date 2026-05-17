"""Build a networkx knowledge graph from the paper database, and answer
cross-literature questions on it.

Edge types
----------
The graph connects papers through several relations, each toggleable:

* ``SHARED_CONCEPT`` — two papers carry the same NLP-derived concept.
* ``SHARED_AUTHOR``  — two papers share an author.
* ``SHARED_AREA``    — two papers sit in the same research area.
* ``CITATION``       — one paper cites another (from the citations table).
* ``SIMILAR_EMBEDDING`` — abstract embeddings above a cosine threshold.

The first three are computed from the join tables; ``CITATION`` from the
citations table; ``SIMILAR_EMBEDDING`` needs ``run_embed_papers.py`` to
have run first.

Cross-literature
----------------
:func:`cross_literature` is the headline analytical query: pick two
research areas and it surfaces the concepts they share and the papers
that bridge them — the "connect points across different literatures"
capability from the brief.

Performance note
----------------
Shared-attribute edges are O(sum of group sizes squared). For a
hundreds-to-low-thousands-of-papers prototype corpus that is fine; for
larger corpora cap group sizes or switch to a bipartite
paper-concept graph projection.
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from xuanzhi.db import Store
from xuanzhi.schema import Edge, EdgeType
from xuanzhi.schema.models import _stable_id

log = logging.getLogger(__name__)

# Which edge families to build by default.
DEFAULT_EDGE_TYPES: tuple[EdgeType, ...] = (
    EdgeType.SHARED_CONCEPT,
    EdgeType.SHARED_AUTHOR,
    EdgeType.SHARED_AREA,
)


# ----------------------------------------------------------- graph build


def build_paper_graph(
    store: Store,
    *,
    edge_types: tuple[EdgeType, ...] = DEFAULT_EDGE_TYPES,
    min_year: int | None = None,
    area_id: str | None = None,
    max_nodes: int | None = 400,
    similarity_threshold: float = 0.55,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> nx.Graph:
    """Build an undirected paper graph.

    Nodes are papers; node attributes include ``title``, ``year``,
    ``source``, ``areas`` and ``citation_count``. Edges carry
    ``edge_type``, ``weight`` and ``evidence``.

    Filters (``min_year``, ``area_id``, ``max_nodes``) are applied to the
    node set *before* edges are computed.
    """
    # ---- node set ---------------------------------------------------------
    papers = list(store.iter_papers())
    if min_year is not None:
        papers = [p for p in papers if (p.year or 0) >= min_year]
    if area_id is not None:
        keep = set(store.paper_ids_for_area(area_id))
        papers = [p for p in papers if p.id in keep]
    if max_nodes is not None and len(papers) > max_nodes:
        # Keep the most-cited papers — they're the interesting hubs.
        papers.sort(key=lambda p: (p.citation_count or 0), reverse=True)
        papers = papers[:max_nodes]

    node_ids = {p.id for p in papers}
    g = nx.Graph()
    for p in papers:
        g.add_node(
            p.id,
            title=p.title,
            year=p.year,
            source=p.source.value,
            citation_count=p.citation_count or 0,
            areas=[a.name for a in p.research_areas],
            url=str(p.url) if p.url else None,
        )

    # ---- shared-attribute edges ------------------------------------------
    if EdgeType.SHARED_CONCEPT in edge_types:
        concept_groups: dict[str, list[str]] = defaultdict(list)
        concept_names: dict[str, str] = {}
        for paper_id, concept in store.iter_all_paper_concepts():
            if paper_id in node_ids:
                concept_groups[concept.id].append(paper_id)
                concept_names[concept.id] = concept.name
        _add_group_edges(
            g, concept_groups, EdgeType.SHARED_CONCEPT, label_map=concept_names
        )

    if EdgeType.SHARED_AUTHOR in edge_types:
        author_groups: dict[str, list[str]] = defaultdict(list)
        author_names: dict[str, str] = {}
        for p in papers:
            for a in p.authors:
                author_groups[a.id].append(p.id)
                author_names[a.id] = a.name
        _add_group_edges(
            g, author_groups, EdgeType.SHARED_AUTHOR, label_map=author_names
        )

    if EdgeType.SHARED_AREA in edge_types:
        area_groups: dict[str, list[str]] = defaultdict(list)
        area_names: dict[str, str] = {}
        for p in papers:
            for area in p.research_areas:
                area_groups[area.id].append(p.id)
                area_names[area.id] = area.name
        _add_group_edges(
            g, area_groups, EdgeType.SHARED_AREA, label_map=area_names
        )

    # ---- citation edges ---------------------------------------------------
    if EdgeType.CITATION in edge_types:
        for cit in store.iter_citations():
            if cit.citing_paper_id in node_ids and cit.cited_paper_id in node_ids:
                _bump_edge(
                    g,
                    cit.citing_paper_id,
                    cit.cited_paper_id,
                    EdgeType.CITATION,
                    evidence="cites",
                )

    # ---- embedding-similarity edges --------------------------------------
    if EdgeType.SIMILAR_EMBEDDING in edge_types:
        _add_similarity_edges(
            g, store, node_ids, embedding_model, similarity_threshold
        )

    log.info(
        "[graph] built: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges()
    )
    return g


def _add_group_edges(
    g: nx.Graph,
    groups: dict[str, list[str]],
    edge_type: EdgeType,
    label_map: dict[str, str],
) -> None:
    """Add a clique of edges for every group that shares an attribute."""
    for key, members in groups.items():
        if len(members) < 2:
            continue
        evidence = label_map.get(key, key)
        for a, b in itertools.combinations(set(members), 2):
            _bump_edge(g, a, b, edge_type, evidence=evidence)


def _bump_edge(
    g: nx.Graph,
    a: str,
    b: str,
    edge_type: EdgeType,
    evidence: str,
) -> None:
    """Add an edge, or strengthen it if a connection already exists.

    Multiple shared attributes between the same two papers accumulate
    into a higher weight — that is exactly the signal we want the graph
    layout to reflect.
    """
    if g.has_edge(a, b):
        data = g[a][b]
        data["weight"] += 1.0
        # Keep a compact, de-duplicated evidence trail.
        ev = set(data.get("evidence", "").split(" | ")) if data.get("evidence") else set()
        ev.add(f"{edge_type.value}:{evidence}")
        data["evidence"] = " | ".join(sorted(e for e in ev if e))
        # "mixed" once more than one edge family is present.
        if data.get("edge_type") != edge_type.value:
            data["edge_type"] = "mixed"
    else:
        g.add_edge(
            a,
            b,
            weight=1.0,
            edge_type=edge_type.value,
            evidence=f"{edge_type.value}:{evidence}",
        )


def _add_similarity_edges(
    g: nx.Graph,
    store: Store,
    node_ids: set[str],
    embedding_model: str,
    threshold: float,
) -> None:
    """Connect papers whose abstract embeddings are close."""
    try:
        import numpy as np

        from xuanzhi.nlp.embeddings import EmbeddingMatrix
    except ImportError:  # pragma: no cover
        log.warning("[graph] numpy/nlp unavailable — skipping similarity edges")
        return

    matrix = EmbeddingMatrix.from_store(store, model=embedding_model)
    if len(matrix) == 0:
        log.info("[graph] no embeddings found — skipping similarity edges")
        return

    # Restrict to nodes actually in the graph.
    idx = [i for i, pid in enumerate(matrix.paper_ids) if pid in node_ids]
    if len(idx) < 2:
        return
    sub_ids = [matrix.paper_ids[i] for i in idx]
    sub_vecs = matrix.vectors[idx]
    sims = sub_vecs @ sub_vecs.T  # cosine: vectors are L2-normalised

    n = len(sub_ids)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            if s >= threshold:
                _bump_edge(
                    g,
                    sub_ids[i],
                    sub_ids[j],
                    EdgeType.SIMILAR_EMBEDDING,
                    evidence=f"cos={s:.2f}",
                )


# ----------------------------------------------------- edge persistence


def persist_derived_edges(store: Store, g: nx.Graph) -> int:
    """Write the graph's edges into the ``edges`` table. Returns the count.

    The UI builds graphs in-memory for responsiveness; this is for when
    you want the derived connections queryable in SQL.
    """
    n = 0
    for a, b, data in g.edges(data=True):
        raw_type = data.get("edge_type", "shared_concept")
        try:
            edge_type = EdgeType(raw_type)
        except ValueError:
            edge_type = EdgeType.SHARED_CONCEPT  # "mixed" falls back here
        edge = Edge(
            id=_stable_id("edge", a, b, raw_type),
            src_paper_id=a,
            dst_paper_id=b,
            edge_type=edge_type,
            weight=float(data.get("weight", 1.0)),
            evidence=data.get("evidence"),
        )
        store.add_edge(edge)
        n += 1
    log.info("[graph] persisted %d edges", n)
    return n


# ------------------------------------------------------ cross-literature


@dataclass
class CrossLiteratureResult:
    """The output of a two-area cross-literature comparison."""

    area_a_name: str
    area_b_name: str
    papers_a: list[str] = field(default_factory=list)  # paper ids
    papers_b: list[str] = field(default_factory=list)
    shared_concepts: list[tuple[str, int]] = field(default_factory=list)  # (name, freq)
    bridging_papers: list[tuple[str, float]] = field(default_factory=list)  # (id, bridge_score)

    @property
    def is_empty(self) -> bool:
        return not self.papers_a or not self.papers_b


def cross_literature(
    store: Store,
    area_a_id: str,
    area_b_id: str,
) -> CrossLiteratureResult:
    """Compare two research areas: what concepts do they share, and which
    papers bridge them?

    A *bridging paper* is a paper in either area that carries at least one
    concept also present in the other area — i.e. it is conceptually
    "speaking to" the other literature.
    """
    areas = {a.id: a for a in store.list_research_areas()}
    area_a = areas.get(area_a_id)
    area_b = areas.get(area_b_id)
    result = CrossLiteratureResult(
        area_a_name=area_a.name if area_a else area_a_id,
        area_b_name=area_b.name if area_b else area_b_id,
    )

    ids_a = set(store.paper_ids_for_area(area_a_id))
    ids_b = set(store.paper_ids_for_area(area_b_id))
    result.papers_a = sorted(ids_a)
    result.papers_b = sorted(ids_b)
    if not ids_a or not ids_b:
        return result

    # Map papers -> concepts and concepts -> their human names.
    paper_concepts: dict[str, set[str]] = defaultdict(set)
    concept_names: dict[str, str] = {}
    for paper_id, concept in store.iter_all_paper_concepts():
        if paper_id in ids_a or paper_id in ids_b:
            paper_concepts[paper_id].add(concept.id)
            concept_names[concept.id] = concept.name

    concepts_a: dict[str, int] = defaultdict(int)
    concepts_b: dict[str, int] = defaultdict(int)
    for pid in ids_a:
        for cid in paper_concepts.get(pid, ()):
            concepts_a[cid] += 1
    for pid in ids_b:
        for cid in paper_concepts.get(pid, ()):
            concepts_b[cid] += 1

    shared_ids = set(concepts_a) & set(concepts_b)
    result.shared_concepts = sorted(
        (
            (concept_names.get(cid, cid), concepts_a[cid] + concepts_b[cid])
            for cid in shared_ids
        ),
        key=lambda t: t[1],
        reverse=True,
    )

    # Bridging papers: carry at least one shared concept, ranked by what fraction
    # of their concepts are cross-area (papers in both areas score 1.0 by definition).
    both = ids_a & ids_b
    bridging_scored: list[tuple[str, float]] = []
    for pid in ids_a | ids_b:
        own = paper_concepts.get(pid, set())
        overlap = own & shared_ids
        if not overlap and pid not in both:
            continue
        if pid in both:
            score = 1.0
        else:
            score = len(overlap) / len(own) if own else 0.0
        bridging_scored.append((pid, round(score, 3)))
    result.bridging_papers = sorted(bridging_scored, key=lambda t: t[1], reverse=True)

    return result
