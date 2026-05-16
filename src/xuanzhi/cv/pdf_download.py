"""Download a paper's PDF, cached on disk.

Kept separate from figure extraction so the (slow, network-bound)
download step is independently cacheable and retryable. Files land in
``data/pdfs/{paper_id}.pdf`` and are reused on subsequent runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

_USER_AGENT = "Xuanzhi/0.1 (academic-research prototype; contact: miguelyu2802@gmail.com)"


def download_pdf(
    pdf_url: str,
    paper_id: str,
    pdf_dir: Path,
    *,
    overwrite: bool = False,
    timeout: float = 60.0,
) -> Path | None:
    """Download ``pdf_url`` to ``pdf_dir/{paper_id}.pdf``.

    Returns the local path, or ``None`` if the download failed. Existing
    files are reused unless ``overwrite=True``.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / f"{paper_id}.pdf"
    if dest.exists() and not overwrite:
        log.debug("[cv] pdf cached: %s", dest.name)
        return dest

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = client.get(pdf_url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not resp.content[:4] == b"%PDF":
                log.warning(
                    "[cv] %s did not return a PDF (content-type=%s)",
                    pdf_url,
                    content_type,
                )
                return None
            dest.write_bytes(resp.content)
            log.info("[cv] downloaded %s (%d KB)", dest.name, len(resp.content) // 1024)
            return dest
    except httpx.HTTPError as e:
        log.warning("[cv] failed to download %s: %s", pdf_url, e)
        return None
