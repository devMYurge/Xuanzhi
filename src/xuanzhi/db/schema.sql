-- Xuanzhi SQLite schema. Mirrors src/xuanzhi/schema/models.py.
--
-- Conventions:
--   * Primary keys are stable hashes (TEXT) computed in Python.
--   * Many-to-many tables have composite PKs and ON DELETE CASCADE so
--     re-ingesting a paper cleanly replaces its related rows.
--   * Timestamps are ISO-8601 strings (TEXT) to keep the file portable.
--
-- Migrations: this file is the canonical schema for v0.1. When we change
-- it, add a new migration file (db/migrations/0002_*.sql) rather than
-- editing this one in place.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ------------------------------------------------------------------ papers

CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    year INTEGER,
    venue TEXT,
    doi TEXT,
    language TEXT DEFAULT 'en',
    url TEXT,
    pdf_url TEXT,
    citation_count INTEGER,
    reference_count INTEGER,
    influential_citation_count INTEGER,
    ingested_at TEXT NOT NULL,
    raw_metadata TEXT,            -- JSON
    UNIQUE (source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_papers_citation_count ON papers(citation_count);

-- ------------------------------------------------------------------ authors

CREATE TABLE IF NOT EXISTS authors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    affiliation TEXT,
    orcid TEXT,
    UNIQUE (normalized_name)
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    author_id TEXT NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    PRIMARY KEY (paper_id, author_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_authors_author ON paper_authors(author_id);

-- ----------------------------------------------------------- research areas

CREATE TABLE IF NOT EXISTS research_areas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    source TEXT NOT NULL,
    UNIQUE (slug, source)
);

CREATE TABLE IF NOT EXISTS paper_areas (
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    area_id TEXT NOT NULL REFERENCES research_areas(id) ON DELETE CASCADE,
    confidence REAL DEFAULT 1.0,
    PRIMARY KEY (paper_id, area_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_areas_area ON paper_areas(area_id);

-- ----------------------------------------------------------------- concepts

CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    extraction_source TEXT NOT NULL,
    UNIQUE (slug, extraction_source)
);

CREATE TABLE IF NOT EXISTS paper_concepts (
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    salience REAL DEFAULT 1.0,
    PRIMARY KEY (paper_id, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_concepts_concept ON paper_concepts(concept_id);

-- ------------------------------------------------------------------ figures

CREATE TABLE IF NOT EXISTS figures (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    page_num INTEGER,
    bbox TEXT,                    -- JSON [x0,y0,x1,y1]
    image_path TEXT NOT NULL,
    caption TEXT,
    figure_type TEXT DEFAULT 'unknown',
    embedding_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_figures_paper ON figures(paper_id);
CREATE INDEX IF NOT EXISTS idx_figures_type ON figures(figure_type);
CREATE INDEX IF NOT EXISTS idx_figures_embedding ON figures(embedding_id);

-- ---------------------------------------------------------------- citations

CREATE TABLE IF NOT EXISTS citations (
    id TEXT PRIMARY KEY,
    citing_paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    cited_paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    context TEXT,
    UNIQUE (citing_paper_id, cited_paper_id)
);

CREATE INDEX IF NOT EXISTS idx_citations_cited ON citations(cited_paper_id);

-- ---------------------------------------------------------------- summaries

CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    UNIQUE (paper_id, model)
);

-- --------------------------------------------------- embeddings (NLP) --
-- One row per (paper, model). Vector is stored as raw bytes from
-- ``numpy.ndarray.tobytes()`` together with ``dim`` so we can reconstruct
-- it with ``np.frombuffer(blob, dtype=np.float32).reshape(-1, dim)``.

CREATE TABLE IF NOT EXISTS paper_embeddings (
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (paper_id, model)
);

CREATE INDEX IF NOT EXISTS idx_paper_embeddings_model ON paper_embeddings(model);

-- One row per (figure, model). Same BLOB convention as paper_embeddings.
-- These are CLIP *image* embeddings — they share a vector space with CLIP
-- text embeddings, which is what makes image-to-source lookup work.

CREATE TABLE IF NOT EXISTS figure_embeddings (
    figure_id TEXT NOT NULL REFERENCES figures(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (figure_id, model)
);

CREATE INDEX IF NOT EXISTS idx_figure_embeddings_model ON figure_embeddings(model);

-- --------------------------------------------------------- edges (graph) --

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    src_paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    dst_paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    evidence TEXT
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_paper_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_paper_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
