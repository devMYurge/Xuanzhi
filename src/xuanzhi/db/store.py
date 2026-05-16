"""SQLite store. Thin wrapper around sqlite3 that takes / returns Pydantic
schema objects so the rest of the codebase doesn't think in SQL.

Why SQLite for the prototype
----------------------------
* Zero-config, single-file, ships with Python.
* Fast enough for the demo corpus (hundreds → low thousands of papers).
* Easy to ship alongside slides as a self-contained artefact.

Why not an ORM
--------------
SQLAlchemy / SQLModel would add a dependency and a learning step the
group doesn't need for a 4-day prototype. Raw SQL is short, explicit,
and the schema is small.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from xuanzhi.schema import (
    Author,
    Citation,
    Concept,
    Edge,
    Figure,
    FigureType,
    Paper,
    PaperArea,
    PaperAuthor,
    PaperConcept,
    ResearchArea,
    Summary,
)

# Schema DDL lives next to this module so it ships as a package resource.
_SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def init_db(db_path: Path) -> None:
    """Create the SQLite file (if missing) and apply the schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA_FILE.read_text(encoding="utf-8"))


class Store:
    """High-level CRUD over the unified schema.

    Usage:
        store = Store("data/xuanzhi.db")
        store.upsert_paper(paper)
        for p in store.iter_papers():
            ...
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        init_db(self.db_path)

    # -------------------------------------------------- connection plumbing

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------- writers

    def upsert_paper(self, paper: Paper) -> str:
        """Insert or update a paper and all its denormalised relations.

        Returns the paper id.
        """
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO papers (
                    id, source, source_id, title, abstract, year, venue, doi,
                    language, url, pdf_url, citation_count, reference_count,
                    influential_citation_count, ingested_at, raw_metadata
                ) VALUES (
                    :id, :source, :source_id, :title, :abstract, :year, :venue, :doi,
                    :language, :url, :pdf_url, :citation_count, :reference_count,
                    :influential_citation_count, :ingested_at, :raw_metadata
                )
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    abstract = COALESCE(excluded.abstract, papers.abstract),
                    year = COALESCE(excluded.year, papers.year),
                    venue = COALESCE(excluded.venue, papers.venue),
                    doi = COALESCE(excluded.doi, papers.doi),
                    url = COALESCE(excluded.url, papers.url),
                    pdf_url = COALESCE(excluded.pdf_url, papers.pdf_url),
                    citation_count = COALESCE(excluded.citation_count, papers.citation_count),
                    reference_count = COALESCE(excluded.reference_count, papers.reference_count),
                    influential_citation_count = COALESCE(
                        excluded.influential_citation_count,
                        papers.influential_citation_count
                    ),
                    raw_metadata = excluded.raw_metadata
                """,
                {
                    "id": paper.id,
                    "source": paper.source.value,
                    "source_id": paper.source_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "venue": paper.venue,
                    "doi": paper.doi,
                    "language": paper.language,
                    "url": str(paper.url) if paper.url else None,
                    "pdf_url": str(paper.pdf_url) if paper.pdf_url else None,
                    "citation_count": paper.citation_count,
                    "reference_count": paper.reference_count,
                    "influential_citation_count": paper.influential_citation_count,
                    "ingested_at": paper.ingested_at.isoformat(),
                    "raw_metadata": json.dumps(paper.raw_metadata, ensure_ascii=False),
                },
            )

            # authors
            for position, a in enumerate(paper.authors):
                c.execute(
                    """
                    INSERT INTO authors (id, name, normalized_name, affiliation, orcid)
                    VALUES (:id, :name, :normalized_name, :affiliation, :orcid)
                    ON CONFLICT(id) DO UPDATE SET
                        affiliation = COALESCE(excluded.affiliation, authors.affiliation),
                        orcid = COALESCE(excluded.orcid, authors.orcid)
                    """,
                    a.model_dump(),
                )
                c.execute(
                    """
                    INSERT OR REPLACE INTO paper_authors (paper_id, author_id, position)
                    VALUES (?, ?, ?)
                    """,
                    (paper.id, a.id, position),
                )

            # research areas
            for area in paper.research_areas:
                c.execute(
                    """
                    INSERT INTO research_areas (id, name, slug, source)
                    VALUES (:id, :name, :slug, :source)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    area.model_dump(),
                )
                c.execute(
                    """
                    INSERT OR IGNORE INTO paper_areas (paper_id, area_id, confidence)
                    VALUES (?, ?, 1.0)
                    """,
                    (paper.id, area.id),
                )

        return paper.id

    def upsert_papers(self, papers: Iterable[Paper]) -> int:
        n = 0
        for p in papers:
            self.upsert_paper(p)
            n += 1
        return n

    def add_concept(self, paper_id: str, concept: Concept, salience: float = 1.0) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO concepts (id, name, slug, extraction_source)
                VALUES (:id, :name, :slug, :extraction_source)
                ON CONFLICT(id) DO NOTHING
                """,
                concept.model_dump(),
            )
            c.execute(
                """
                INSERT OR REPLACE INTO paper_concepts (paper_id, concept_id, salience)
                VALUES (?, ?, ?)
                """,
                (paper_id, concept.id, salience),
            )

    def add_figure(self, fig: Figure) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO figures
                    (id, paper_id, page_num, bbox, image_path, caption, figure_type, embedding_id)
                VALUES
                    (:id, :paper_id, :page_num, :bbox, :image_path, :caption, :figure_type, :embedding_id)
                """,
                {
                    "id": fig.id,
                    "paper_id": fig.paper_id,
                    "page_num": fig.page_num,
                    "bbox": json.dumps(list(fig.bbox)) if fig.bbox else None,
                    "image_path": fig.image_path,
                    "caption": fig.caption,
                    "figure_type": fig.figure_type.value,
                    "embedding_id": fig.embedding_id,
                },
            )

    def get_figure(self, figure_id: str) -> Figure | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM figures WHERE id = ?", (figure_id,)
            ).fetchone()
            return self._row_to_figure(row) if row else None

    def iter_figures(self, paper_id: str | None = None) -> Iterator[Figure]:
        """Yield figures, optionally filtered to a single paper."""
        with self._conn() as c:
            if paper_id is None:
                cur = c.execute("SELECT * FROM figures")
            else:
                cur = c.execute(
                    "SELECT * FROM figures WHERE paper_id = ?", (paper_id,)
                )
            for row in cur:
                yield self._row_to_figure(row)

    def count_figures(self) -> int:
        with self._conn() as c:
            (n,) = c.execute("SELECT COUNT(*) FROM figures").fetchone()
            return n

    def put_figure_embedding(
        self, figure_id: str, model: str, vector: bytes, dim: int
    ) -> None:
        """Persist a CLIP image embedding for a figure (see put_embedding)."""
        from datetime import datetime, timezone

        with self._conn() as c:
            c.execute(
                """
                INSERT INTO figure_embeddings (figure_id, model, dim, vector, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(figure_id, model) DO UPDATE SET
                    dim = excluded.dim,
                    vector = excluded.vector,
                    created_at = excluded.created_at
                """,
                (
                    figure_id,
                    model,
                    dim,
                    vector,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def iter_figure_embeddings(
        self, model: str
    ) -> Iterator[tuple[str, int, bytes]]:
        """Yield ``(figure_id, dim, vector_bytes)`` for every figure embedding."""
        with self._conn() as c:
            for row in c.execute(
                "SELECT figure_id, dim, vector FROM figure_embeddings WHERE model = ?",
                (model,),
            ):
                yield row["figure_id"], row["dim"], row["vector"]

    def figure_ids_missing_embedding(self, model: str) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT f.id FROM figures f
                LEFT JOIN figure_embeddings e
                       ON e.figure_id = f.id AND e.model = ?
                WHERE e.figure_id IS NULL
                """,
                (model,),
            ).fetchall()
            return [r["id"] for r in rows]

    def add_citation(self, cit: Citation) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT OR IGNORE INTO citations (id, citing_paper_id, cited_paper_id, context)
                VALUES (?, ?, ?, ?)
                """,
                (cit.id, cit.citing_paper_id, cit.cited_paper_id, cit.context),
            )

    def add_summary(self, summary: Summary) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO summaries (id, paper_id, model, summary_text, generated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(paper_id, model) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    generated_at = excluded.generated_at
                """,
                (
                    summary.id,
                    summary.paper_id,
                    summary.model,
                    summary.summary_text,
                    summary.generated_at.isoformat(),
                ),
            )

    # -------------------------------------------- embeddings (NLP) layer

    def put_embedding(
        self,
        paper_id: str,
        model: str,
        vector: bytes,
        dim: int,
    ) -> None:
        """Persist a single paper's embedding vector.

        ``vector`` is expected to be ``np.ndarray.tobytes()`` of a 1-D
        float32 array of length ``dim``. The NLP module owns the numpy
        side; this layer is byte-agnostic so the DB stays decoupled from
        numpy at import time.
        """
        from datetime import datetime, timezone

        with self._conn() as c:
            c.execute(
                """
                INSERT INTO paper_embeddings (paper_id, model, dim, vector, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(paper_id, model) DO UPDATE SET
                    dim = excluded.dim,
                    vector = excluded.vector,
                    created_at = excluded.created_at
                """,
                (
                    paper_id,
                    model,
                    dim,
                    vector,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def iter_embeddings(self, model: str) -> Iterator[tuple[str, int, bytes]]:
        """Yield ``(paper_id, dim, vector_bytes)`` for every embedding under ``model``."""
        with self._conn() as c:
            for row in c.execute(
                "SELECT paper_id, dim, vector FROM paper_embeddings WHERE model = ?",
                (model,),
            ):
                yield row["paper_id"], row["dim"], row["vector"]

    def paper_ids_missing_embedding(self, model: str) -> list[str]:
        """Return ids of papers with no embedding under ``model`` yet."""
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT p.id FROM papers p
                LEFT JOIN paper_embeddings e
                       ON e.paper_id = p.id AND e.model = ?
                WHERE e.paper_id IS NULL
                """,
                (model,),
            ).fetchall()
            return [r["id"] for r in rows]

    def add_edge(self, edge: Edge) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO edges
                    (id, src_paper_id, dst_paper_id, edge_type, weight, evidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.id,
                    edge.src_paper_id,
                    edge.dst_paper_id,
                    edge.edge_type.value,
                    edge.weight,
                    edge.evidence,
                ),
            )

    # ------------------------------------------------------------- readers

    def get_paper(self, paper_id: str) -> Paper | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
            if row is None:
                return None
            return self._row_to_paper(c, row)

    def iter_papers(self, limit: int | None = None) -> Iterator[Paper]:
        with self._conn() as c:
            sql = "SELECT * FROM papers ORDER BY ingested_at DESC"
            if limit is not None:
                sql += f" LIMIT {int(limit)}"
            for row in c.execute(sql):
                yield self._row_to_paper(c, row)

    def count_papers(self) -> int:
        with self._conn() as c:
            (n,) = c.execute("SELECT COUNT(*) FROM papers").fetchone()
            return n

    # ------------------------------------------------ graph-support readers

    def get_concepts_for_paper(self, paper_id: str) -> list[tuple[Concept, float]]:
        """Return ``[(Concept, salience)]`` attached to a paper."""
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT co.*, pc.salience FROM concepts co
                JOIN paper_concepts pc ON pc.concept_id = co.id
                WHERE pc.paper_id = ?
                ORDER BY pc.salience DESC
                """,
                (paper_id,),
            ).fetchall()
            return [
                (
                    Concept(
                        id=r["id"],
                        name=r["name"],
                        slug=r["slug"],
                        extraction_source=r["extraction_source"],
                    ),
                    r["salience"],
                )
                for r in rows
            ]

    def iter_all_paper_concepts(self) -> Iterator[tuple[str, Concept]]:
        """Yield ``(paper_id, Concept)`` for every paper-concept link."""
        with self._conn() as c:
            for r in c.execute(
                """
                SELECT pc.paper_id, co.* FROM concepts co
                JOIN paper_concepts pc ON pc.concept_id = co.id
                """
            ):
                yield r["paper_id"], Concept(
                    id=r["id"],
                    name=r["name"],
                    slug=r["slug"],
                    extraction_source=r["extraction_source"],
                )

    def iter_citations(self) -> Iterator[Citation]:
        with self._conn() as c:
            for r in c.execute("SELECT * FROM citations"):
                yield Citation(
                    id=r["id"],
                    citing_paper_id=r["citing_paper_id"],
                    cited_paper_id=r["cited_paper_id"],
                    context=r["context"],
                )

    def list_research_areas(self) -> list[ResearchArea]:
        """All research areas, with the paper count appended to each name's
        usefulness left to the caller (counts via :meth:`area_paper_counts`).
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM research_areas ORDER BY name"
            ).fetchall()
            return [
                ResearchArea(
                    id=r["id"], name=r["name"], slug=r["slug"], source=r["source"]
                )
                for r in rows
            ]

    def list_concepts(self) -> list[Concept]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM concepts ORDER BY name").fetchall()
            return [
                Concept(
                    id=r["id"],
                    name=r["name"],
                    slug=r["slug"],
                    extraction_source=r["extraction_source"],
                )
                for r in rows
            ]

    def area_paper_counts(self) -> dict[str, int]:
        """Return ``{area_id: paper_count}``."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT area_id, COUNT(*) AS n FROM paper_areas GROUP BY area_id"
            ).fetchall()
            return {r["area_id"]: r["n"] for r in rows}

    def paper_ids_for_area(self, area_id: str) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT paper_id FROM paper_areas WHERE area_id = ?", (area_id,)
            ).fetchall()
            return [r["paper_id"] for r in rows]

    def get_summaries_for_paper(self, paper_id: str) -> list[Summary]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM summaries WHERE paper_id = ?", (paper_id,)
            ).fetchall()
            return [
                Summary(
                    id=r["id"],
                    paper_id=r["paper_id"],
                    model=r["model"],
                    summary_text=r["summary_text"],
                    generated_at=r["generated_at"],
                )
                for r in rows
            ]

    def source_counts(self) -> dict[str, int]:
        """Return ``{source: paper_count}`` for the Overview dashboard."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT source, COUNT(*) AS n FROM papers GROUP BY source"
            ).fetchall()
            return {r["source"]: r["n"] for r in rows}

    # ------------------------------------------------------------ internals

    def _row_to_figure(self, row: sqlite3.Row) -> Figure:
        bbox = json.loads(row["bbox"]) if row["bbox"] else None
        return Figure(
            id=row["id"],
            paper_id=row["paper_id"],
            page_num=row["page_num"],
            bbox=tuple(bbox) if bbox else None,
            image_path=row["image_path"],
            caption=row["caption"],
            figure_type=FigureType(row["figure_type"] or "unknown"),
            embedding_id=row["embedding_id"],
        )

    def _row_to_paper(self, c: sqlite3.Connection, row: sqlite3.Row) -> Paper:
        authors = [
            Author(
                id=r["id"],
                name=r["name"],
                normalized_name=r["normalized_name"],
                affiliation=r["affiliation"],
                orcid=r["orcid"],
            )
            for r in c.execute(
                """
                SELECT a.* FROM authors a
                JOIN paper_authors pa ON pa.author_id = a.id
                WHERE pa.paper_id = ?
                ORDER BY pa.position
                """,
                (row["id"],),
            ).fetchall()
        ]
        areas = [
            ResearchArea(id=r["id"], name=r["name"], slug=r["slug"], source=r["source"])
            for r in c.execute(
                """
                SELECT ra.* FROM research_areas ra
                JOIN paper_areas pra ON pra.area_id = ra.id
                WHERE pra.paper_id = ?
                """,
                (row["id"],),
            ).fetchall()
        ]
        return Paper(
            id=row["id"],
            source=row["source"],
            source_id=row["source_id"],
            title=row["title"],
            abstract=row["abstract"],
            year=row["year"],
            venue=row["venue"],
            doi=row["doi"],
            language=row["language"] or "en",
            url=row["url"],
            pdf_url=row["pdf_url"],
            authors=authors,
            research_areas=areas,
            citation_count=row["citation_count"],
            reference_count=row["reference_count"],
            influential_citation_count=row["influential_citation_count"],
            ingested_at=row["ingested_at"],
            raw_metadata=json.loads(row["raw_metadata"] or "{}"),
        )
