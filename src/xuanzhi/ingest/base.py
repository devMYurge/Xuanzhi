"""Base contract for ingest sources.

Every ingest module (Playwright-scraped or REST-API) implements
``IngestSource`` so the orchestrator can treat them uniformly. This is
how we keep the schema invariant across heterogeneous sources.
"""

from __future__ import annotations

import abc
import asyncio
import random
from typing import AsyncIterator

from xuanzhi.schema import Paper


class IngestSource(abc.ABC):
    """Abstract source. Async so Playwright + httpx fit naturally."""

    #: Human-friendly source name (used in logs).
    name: str = "unknown"

    @abc.abstractmethod
    async def search(self, query: str, max_results: int = 25) -> AsyncIterator[Paper]:
        """Yield Paper objects matching ``query``.

        Implementations should yield as they discover, not as a list, so
        the consumer can stream into the DB.
        """
        raise NotImplementedError
        if False:  # pragma: no cover — make this a real async generator
            yield  # noqa: E702


async def polite_delay(min_s: float = 1.0, max_s: float = 3.0) -> None:
    """Async equivalent of the v0's ``human_like_delay``.

    Used between requests to a single host so we don't hammer it. Per
    ArXiv's ToS the rate of crawl should be modest (~one request every
    few seconds) — these defaults stay well inside that envelope.
    """
    await asyncio.sleep(random.uniform(min_s, max_s))
