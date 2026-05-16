"""Semantic Scholar Graph API client.

Why an API (not Playwright) for this source
-------------------------------------------
Semantic Scholar ships a generous, free public Graph API with everything
we need: paper search, paper lookup, references, citations, embeddings.
Scraping it would be pure busywork. The point of using Playwright is to
demonstrate the technique against sources that *don't* have an API
(Google Scholar, journal landing pages); for Semantic Scholar we go
straight to JSON.

Set ``SEMANTIC_SCHOLAR_API_KEY`` in the environment for higher rate limits.

Docs: https://api.semanticscholar.org/api-docs/graph
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator

import httpx

from xuanzhi.schema import Author, Paper, ResearchArea, Source

from .base import IngestSource, polite_delay

log = logging.getLogger(__name__)

_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "abstract",
        "year",
        "venue",
        "externalIds",
        "openAccessPdf",
        "authors.authorId",
        "authors.name",
        "authors.affiliations",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "citationCount",
        "referenceCount",
        "influentialCitationCount",
    ]
)


class SemanticScholarSource(IngestSource):
    """REST-based ingest for Semantic Scholar Graph API."""

    name = "semantic_scholar"

    def __init__(self, api_key: str | None = None, polite_s: float = 1.0):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.polite_s = polite_s
        headers = {"User-Agent": "Xuanzhi/0.1 (academic-research prototype)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self._headers = headers

    # ----------------------------------------------------------- public

    async def search(
        self,
        query: str,
        max_results: int = 25,
    ) -> AsyncIterator[Paper]:
        """Paged paper search."""
        async with httpx.AsyncClient(timeout=30.0, headers=self._headers) as client:
            offset = 0
            yielded = 0
            page_size = min(100, max_results)
            while yielded < max_results:
                params = {
                    "query": query,
                    "offset": offset,
                    "limit": page_size,
                    "fields": _FIELDS,
                }
                resp = await self._get_with_retry(
                    client, f"{_BASE}/paper/search", params
                )
                data = resp.json()
                hits = data.get("data") or []
                if not hits:
                    return
                for raw in hits:
                    paper = self._row_to_paper(raw)
                    if paper is not None:
                        yield paper
                        yielded += 1
                        if yielded >= max_results:
                            return
                offset = data.get("next") or (offset + len(hits))
                await polite_delay(self.polite_s, self.polite_s * 2)

    async def enrich(self, paper_id: str) -> Paper | None:
        """Fetch a single paper by Semantic Scholar id or any external id.

        Accepts ``ARXIV:2401.12345``, ``DOI:10.xxx``, ``CorpusId:...``, or
        a raw Semantic Scholar paperId.
        """
        async with httpx.AsyncClient(timeout=30.0, headers=self._headers) as client:
            resp = await self._get_with_retry(
                client, f"{_BASE}/paper/{paper_id}", {"fields": _FIELDS}
            )
            return self._row_to_paper(resp.json())

    # ----------------------------------------------------------- helpers

    async def _get_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict,
        max_attempts: int = 5,
    ) -> httpx.Response:
        backoff = 1.0
        last: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.get(url, params=params)
                if resp.status_code == 429:
                    log.warning("[s2] 429 rate-limited — backoff %.1fs (attempt %d/%d)", backoff, attempt, max_attempts)
                    last = RuntimeError(
                        "Semantic Scholar is rate-limiting this IP. "
                        "Register for a free API key at https://www.semanticscholar.org/product/api "
                        "and set SEMANTIC_SCHOLAR_API_KEY in your .env file."
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                return resp
            except httpx.HTTPError as e:
                last = e
                log.warning("[s2] attempt %d failed: %s", attempt, e)
                await asyncio.sleep(backoff)
                backoff *= 2
        raise RuntimeError(str(last))

    def _row_to_paper(self, r: dict) -> Paper | None:
        paper_id = r.get("paperId")
        title = (r.get("title") or "").strip()
        if not paper_id or not title:
            return None

        author_objs: list[Author] = []
        for a in r.get("authors") or []:
            name = (a.get("name") or "").strip()
            if not name:
                continue
            affs = a.get("affiliations") or []
            author_objs.append(
                Author.from_name(name, affiliation=", ".join(affs) if affs else None)
            )

        # fieldsOfStudy is a flat list; s2FieldsOfStudy is richer.
        areas: list[ResearchArea] = []
        seen: set[str] = set()
        for name in r.get("fieldsOfStudy") or []:
            if name and name not in seen:
                seen.add(name)
                areas.append(ResearchArea.make(name, source="s2-fieldOfStudy"))
        for f in r.get("s2FieldsOfStudy") or []:
            name = f.get("category") if isinstance(f, dict) else None
            if name and name not in seen:
                seen.add(name)
                areas.append(ResearchArea.make(name, source="s2-fieldOfStudy"))

        external = r.get("externalIds") or {}
        doi = external.get("DOI")
        pdf_url = (r.get("openAccessPdf") or {}).get("url") or None
        url = f"https://www.semanticscholar.org/paper/{paper_id}"

        return Paper(
            id=Paper.build_id(Source.SEMANTIC_SCHOLAR, paper_id),
            source=Source.SEMANTIC_SCHOLAR,
            source_id=paper_id,
            title=title,
            abstract=r.get("abstract"),
            year=r.get("year"),
            venue=r.get("venue"),
            doi=doi,
            url=url,
            pdf_url=pdf_url,
            authors=author_objs,
            research_areas=areas,
            citation_count=r.get("citationCount"),
            reference_count=r.get("referenceCount"),
            influential_citation_count=r.get("influentialCitationCount"),
            raw_metadata={"externalIds": external},
        )
