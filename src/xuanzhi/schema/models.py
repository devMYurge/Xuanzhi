"""Unified data model for Xuanzhi.

Designed for a literature-review use case across heterogeneous sources
(ArXiv, Semantic Scholar, OpenAlex, Google Scholar, ...). Every ingest
module must emit objects that conform to this schema so downstream NLP,
CV, graph and UI code can stay source-agnostic.

Design notes
------------
- IDs are stable hashes of (source, source_id) so re-runs are idempotent
  and the same paper from different sources can be reconciled later.
- Optional everywhere defaults are sensible — early ingest emits sparse
  records; enrichment passes (citations, concepts, figures) fill them in.
- The Edge model is a free-form connection between papers — citation,
  shared concept, shared author, embedding similarity, etc. The graph
  module derives edges; ingest is not responsible for them.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


# ---------- enums -----------------------------------------------------------


class Source(str, Enum):
    """Where a paper record originated."""

    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"
    GOOGLE_SCHOLAR = "google_scholar"
    PUBMED = "pubmed"
    BIORXIV = "biorxiv"
    MANUAL = "manual"


class FigureType(str, Enum):
    """CV-classified figure types (assigned by xuanzhi.cv)."""

    CHART = "chart"
    DIAGRAM = "diagram"
    PHOTO = "photo"
    TABLE = "table"
    EQUATION = "equation"
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    """How two papers (or a paper and a concept) are connected in the graph."""

    CITATION = "citation"
    SHARED_AUTHOR = "shared_author"
    SHARED_AREA = "shared_area"
    SHARED_CONCEPT = "shared_concept"
    SIMILAR_EMBEDDING = "similar_embedding"
    BRIDGES_LITERATURES = "bridges_literatures"


# ---------- helpers ---------------------------------------------------------


def _stable_id(*parts: str) -> str:
    """Deterministic short ID for cross-source reconciliation."""
    raw = "|".join(p.strip().lower() for p in parts if p)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ---------- core entities ---------------------------------------------------


class Author(BaseModel):
    """A paper author. ``normalized_name`` is used for deduplication."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    normalized_name: str
    affiliation: str | None = None
    orcid: str | None = None

    @classmethod
    def from_name(cls, name: str, affiliation: str | None = None) -> "Author":
        norm = " ".join(name.lower().split())
        return cls(
            id=_stable_id("author", norm),
            name=name.strip(),
            normalized_name=norm,
            affiliation=affiliation,
        )


class ResearchArea(BaseModel):
    """A high-level field tag, e.g. an ArXiv category or OpenAlex concept."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    slug: str
    source: str  # "arxiv-category" | "openalex-concept" | "user" | ...

    @classmethod
    def make(cls, name: str, source: str) -> "ResearchArea":
        slug = "-".join(name.lower().split())
        return cls(
            id=_stable_id("area", source, slug),
            name=name,
            slug=slug,
            source=source,
        )


class Concept(BaseModel):
    """A finer-grained extracted concept (NER, topic model, cluster centroid)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    slug: str
    extraction_source: str  # "ner" | "cluster" | "topic-model" | "manual"

    @classmethod
    def make(cls, name: str, extraction_source: str) -> "Concept":
        slug = "-".join(name.lower().split())
        return cls(
            id=_stable_id("concept", extraction_source, slug),
            name=name,
            slug=slug,
            extraction_source=extraction_source,
        )


class Figure(BaseModel):
    """A figure extracted from a paper PDF, with provenance back to its source."""

    model_config = ConfigDict(extra="ignore")

    id: str
    paper_id: str
    page_num: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    image_path: str  # local path or URL to the extracted image
    caption: str | None = None
    figure_type: FigureType = FigureType.UNKNOWN
    embedding_id: str | None = None  # FAISS / annoy / sklearn index ref


class Citation(BaseModel):
    """A directed citation from one paper to another."""

    model_config = ConfigDict(extra="ignore")

    id: str
    citing_paper_id: str
    cited_paper_id: str
    context: str | None = None  # the sentence around the citation, if known


class Summary(BaseModel):
    """A summary of a paper. Multiple summaries per paper allowed (one per model)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    paper_id: str
    model: str  # "gpt-4o-mini" | "facebook/bart-large-cnn" | ...
    summary_text: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------- join / link tables ----------------------------------------------


class PaperAuthor(BaseModel):
    paper_id: str
    author_id: str
    position: int  # author order, 0-indexed


class PaperArea(BaseModel):
    paper_id: str
    area_id: str
    confidence: float = 1.0  # 1.0 if from source metadata; <1 if classifier


class PaperConcept(BaseModel):
    paper_id: str
    concept_id: str
    salience: float = 1.0


# ---------- edges (graph layer) ---------------------------------------------


class Edge(BaseModel):
    """A connection in the cross-literature knowledge graph.

    Most edges are derived by :mod:`xuanzhi.graph`, not by ingest. Stored
    persistently so the UI can browse without recomputing.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    src_paper_id: str
    dst_paper_id: str
    edge_type: EdgeType
    weight: float = 1.0
    evidence: str | None = None  # e.g. shared concept name, similarity score


# ---------- the main entity -------------------------------------------------


class Paper(BaseModel):
    """A research paper. The hub of the schema.

    A paper's ``id`` is a hash of (source, source_id) so the same paper
    ingested from different sources will not collide, and the same paper
    re-ingested will be idempotent.
    """

    model_config = ConfigDict(extra="ignore")

    # identity
    id: str
    source: Source
    source_id: str  # arxiv id, ss paper id, openalex id, doi, etc.

    # bibliographic
    title: str
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    language: str = "en"

    # links
    url: HttpUrl | None = None
    pdf_url: HttpUrl | None = None

    # related entities (denormalised for convenience; canonical lives in DB)
    authors: list[Author] = Field(default_factory=list)
    research_areas: list[ResearchArea] = Field(default_factory=list)

    # bibliometrics (filled by Semantic Scholar / OpenAlex enrichment)
    citation_count: int | None = None
    reference_count: int | None = None
    influential_citation_count: int | None = None

    # provenance
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "abstract", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @classmethod
    def build_id(cls, source: Source, source_id: str) -> str:
        return _stable_id(source.value, source_id)
