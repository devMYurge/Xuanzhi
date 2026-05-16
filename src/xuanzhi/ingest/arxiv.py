"""ArXiv ingest — Playwright port of the v0 Selenium scraper.

Why Playwright (over Selenium)
------------------------------
* Auto-wait built into every action (no manual ``WebDriverWait`` chains).
* First-class async API — fits ``asyncio`` for concurrent multi-source runs.
* Network interception lets us snap PDF URLs cheaply.
* Single ``playwright install`` provisions Chromium/Firefox/WebKit; no
  ``webdriver-manager`` drift.

What it does
------------
* Hits ``https://arxiv.org/search/`` with the query.
* Walks paginated result cards (``li.arxiv-result``), extracts:
    - arXiv id (the canonical ``source_id``)
    - title, authors, abstract, primary + cross-list categories
    - PDF link
* Emits :class:`xuanzhi.schema.Paper` objects asynchronously.

Etiquette
---------
* ``polite_delay`` between page loads (ArXiv ToS asks for modest crawl
  rates).
* Realistic user-agent set on the browser context.
* Headless by default; pass ``headless=False`` while developing to watch
  the run.

Limitations
-----------
* Citation count is **not** on ArXiv; enrich via Semantic Scholar.
* Cross-listed categories are captured but not deduped against primary.
* For literal high-volume ingestion the OAI-PMH or ATOM API is faster —
  but this Playwright path is the AIML-class deliverable and the
  pattern reuses straightforwardly for sources that *don't* have an API.
"""

from __future__ import annotations

import logging
import re
from typing import AsyncIterator
from urllib.parse import urlencode

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    async_playwright,
)

from xuanzhi.schema import Author, Paper, ResearchArea, Source

from .base import IngestSource, polite_delay

log = logging.getLogger(__name__)

_SEARCH_URL = "https://arxiv.org/search/"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 "
    "Xuanzhi/0.1 (academic-research prototype; contact: miguelyu2802@gmail.com)"
)
_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/([\w.\-/]+)", re.IGNORECASE)


class ArxivPlaywrightSource(IngestSource):
    """Playwright-driven ArXiv scraper, async-iterator interface."""

    name = "arxiv"

    def __init__(
        self,
        headless: bool = True,
        per_page: int = 25,
        polite_min: float = 2.0,
        polite_max: float = 4.5,
    ):
        self.headless = headless
        self.per_page = max(25, min(per_page, 200))  # ArXiv allows 25/50/100/200
        self.polite_min = polite_min
        self.polite_max = polite_max

    # ------------------------------------------------------ public API

    async def search(
        self,
        query: str,
        max_results: int = 25,
    ) -> AsyncIterator[Paper]:
        """Yield ArXiv papers matching ``query``.

        ``query`` is passed straight through to ArXiv's full-text search.
        Stops once ``max_results`` papers have been yielded.
        """
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self.headless)
            try:
                context = await browser.new_context(user_agent=_USER_AGENT)
                page = await context.new_page()
                yielded = 0
                start = 0
                while yielded < max_results:
                    url = self._build_url(query, start=start)
                    log.info("[arxiv] GET %s", url)
                    await page.goto(url, wait_until="domcontentloaded")
                    # The list of result cards is the auto-wait target.
                    await page.wait_for_selector(
                        "li.arxiv-result, p.is-size-4",
                        timeout=15_000,
                    )

                    papers = await self._extract_results(page)
                    if not papers:
                        log.info("[arxiv] no more results at offset %d", start)
                        break

                    for paper in papers:
                        yield paper
                        yielded += 1
                        if yielded >= max_results:
                            break

                    start += self.per_page
                    await polite_delay(self.polite_min, self.polite_max)
            finally:
                await browser.close()

    # ------------------------------------------------------ internals

    def _build_url(self, query: str, start: int) -> str:
        params = {
            "searchtype": "all",
            "query": query,
            "start": start,
            "size": self.per_page,
        }
        return f"{_SEARCH_URL}?{urlencode(params)}"

    async def _extract_results(self, page: Page) -> list[Paper]:
        """Pull every ``li.arxiv-result`` on the current page."""
        # We extract via a single page.evaluate so the round-trips don't
        # accumulate latency per field. This is also more robust than
        # Selenium's element-by-element walk.
        raw: list[dict] = await page.evaluate(
            """
            () => Array.from(document.querySelectorAll('li.arxiv-result')).map(li => {
                const text = sel => {
                    const n = li.querySelector(sel);
                    return n ? n.innerText.trim() : null;
                };
                const href = sel => {
                    const n = li.querySelector(sel);
                    return n ? n.getAttribute('href') : null;
                };
                const idAnchor = li.querySelector('p.list-title a');
                const idHref = idAnchor ? idAnchor.getAttribute('href') : null;
                const title = text('p.title');
                const abstractFull = li.querySelector('p.abstract span.abstract-full');
                const abstractShort = li.querySelector('p.abstract span.abstract-short');
                const abstract = (abstractFull && abstractFull.innerText.trim())
                    || (abstractShort && abstractShort.innerText.trim())
                    || null;
                const authors = Array.from(li.querySelectorAll('p.authors a'))
                    .map(a => a.innerText.trim())
                    .filter(Boolean);
                const primaryTag = li.querySelector('p.tags span.primary-subject');
                const allTags = Array.from(li.querySelectorAll('p.tags span.tag'))
                    .map(t => t.innerText.trim())
                    .filter(Boolean);
                const pdfLink = href('p.list-title a[href*="/pdf/"]');
                const submittedRaw = text('p.is-size-7');
                return {
                    idHref,
                    title,
                    abstract,
                    authors,
                    primaryTag: primaryTag ? primaryTag.innerText.trim() : null,
                    allTags,
                    pdfLink,
                    submittedRaw,
                };
            })
            """
        )

        papers: list[Paper] = []
        for r in raw:
            paper = self._row_to_paper(r)
            if paper is not None:
                papers.append(paper)
        return papers

    def _row_to_paper(self, r: dict) -> Paper | None:
        id_href = r.get("idHref") or ""
        m = _ARXIV_ID_RE.search(id_href)
        if not m:
            log.warning("[arxiv] dropping row without parseable id: %r", id_href)
            return None
        arxiv_id = m.group(1)

        title = (r.get("title") or "").strip()
        if not title:
            return None

        # Strip the "Title: " prefix arXiv puts in p.title.innerText.
        title = re.sub(r"^Title:\s*", "", title)

        # Authors → schema objects
        author_objs = [Author.from_name(a) for a in (r.get("authors") or [])]

        # Areas: primary tag is canonical, plus any cross-list tags.
        areas: list[ResearchArea] = []
        seen: set[str] = set()
        for tag in filter(None, [r.get("primaryTag"), *r.get("allTags", [])]):
            if tag in seen:
                continue
            seen.add(tag)
            areas.append(ResearchArea.make(tag, source="arxiv-category"))

        year = _parse_year(r.get("submittedRaw"))

        pdf_url = r.get("pdfLink")
        if pdf_url and pdf_url.startswith("/"):
            pdf_url = f"https://arxiv.org{pdf_url}"

        abs_url = f"https://arxiv.org/abs/{arxiv_id}"

        return Paper(
            id=Paper.build_id(Source.ARXIV, arxiv_id),
            source=Source.ARXIV,
            source_id=arxiv_id,
            title=title,
            abstract=r.get("abstract"),
            year=year,
            url=abs_url,
            pdf_url=pdf_url,
            authors=author_objs,
            research_areas=areas,
            raw_metadata={
                "submitted_raw": r.get("submittedRaw"),
                "primary_tag": r.get("primaryTag"),
                "all_tags": r.get("allTags"),
            },
        )


def _parse_year(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", text)
    return int(m.group(0)) if m else None
