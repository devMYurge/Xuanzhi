"""Ingest modules. Each source emits Paper objects under the unified schema.

Sources currently wired up:
    * arxiv             — Playwright scrape of arxiv.org/search
    * semantic_scholar  — REST API client (citations, references, enrichment)

Sources stubbed for follow-up:
    * google_scholar    — Playwright (anti-bot heavy; lower priority)
    * openalex          — REST API (good for cross-discipline areas)
    * pubmed / biorxiv  — REST APIs
"""

from .base import IngestSource
from .arxiv import ArxivPlaywrightSource
from .semantic_scholar import SemanticScholarSource

__all__ = ["IngestSource", "ArxivPlaywrightSource", "SemanticScholarSource"]
