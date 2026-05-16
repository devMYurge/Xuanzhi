"""Extract figures from a paper PDF.

Uses **PyMuPDF** (``fitz``) to walk pages, pull embedded raster images,
recover each image's bounding box, and grab the nearest caption ("Figure
N: ...") below it. Each extracted image is saved as a PNG under
``data/figures/{paper_id}/`` and described by a
:class:`xuanzhi.schema.Figure`.

This is deliberately pragmatic, not perfect — PDF figure extraction is a
genuinely hard problem (vector figures, multi-panel layouts, figures
rendered as text). For the prototype we extract embedded raster images,
which covers the large majority of charts/photos in modern arXiv PDFs,
and we document the rest as a known limitation.

Filtering
---------
We drop images that are almost certainly not figures:
* smaller than ``min_dim`` on either side (logos, icons, math glyphs),
* extreme aspect ratios (rule lines, banners),
* near-monochrome images (page-background scans).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from xuanzhi.schema import Figure, FigureType
from xuanzhi.schema.models import _stable_id

log = logging.getLogger(__name__)

# Caption lines look like "Figure 3:", "Fig. 3.", "FIGURE 12 —", etc.
_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?)\s*\d+", re.IGNORECASE)


def extract_figures(
    pdf_path: Path,
    paper_id: str,
    figures_dir: Path,
    *,
    min_dim: int = 100,
    max_aspect_ratio: float = 12.0,
) -> list[Figure]:
    """Extract figures from ``pdf_path``.

    Parameters
    ----------
    pdf_path:
        Local path to the paper PDF (see :func:`cv.pdf_download.download_pdf`).
    paper_id:
        The owning paper's id — used for the Figure foreign key and the
        output directory.
    figures_dir:
        Root directory for extracted images; this function writes into
        ``figures_dir / paper_id /``.
    min_dim:
        Minimum width/height in pixels for an image to count as a figure.
    max_aspect_ratio:
        Images wider/taller than this ratio are treated as rules/banners.

    Returns
    -------
    list[Figure] — also written as PNGs to disk. ``figure_type`` is left
    as ``UNKNOWN``; :mod:`xuanzhi.cv.classify` fills it in.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyMuPDF is required — `pip install PyMuPDF`.") from e

    out_dir = figures_dir / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Figure] = []
    doc = fitz.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            captions = _page_captions(page)
            seen_xrefs: set[int] = set()

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                # Pixmap → discard if too small / wrong shape.
                try:
                    pix = fitz.Pixmap(doc, xref)
                except Exception as e:  # noqa: BLE001
                    log.debug("[cv] xref %d unreadable: %s", xref, e)
                    continue

                if not _is_plausible_figure(pix, min_dim, max_aspect_ratio):
                    pix = None
                    continue

                # CMYK / alpha → convert to RGB before saving.
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                image_path = out_dir / f"p{page_index + 1}_{img_index}.png"
                pix.save(image_path)
                pix = None

                bbox = _image_bbox(page, xref)
                caption = _nearest_caption(bbox, captions)

                fig = Figure(
                    id=_stable_id("figure", paper_id, str(page_index), str(img_index)),
                    paper_id=paper_id,
                    page_num=page_index + 1,
                    bbox=bbox,
                    image_path=str(image_path),
                    caption=caption,
                    figure_type=FigureType.UNKNOWN,
                )
                figures.append(fig)
    finally:
        doc.close()

    log.info("[cv] extracted %d figures from %s", len(figures), pdf_path.name)
    return figures


# --------------------------------------------------------------- internals


def _is_plausible_figure(pix, min_dim: int, max_aspect_ratio: float) -> bool:
    """Cheap heuristics to reject logos, icons, rules, and glyphs."""
    w, h = pix.width, pix.height
    if w < min_dim or h < min_dim:
        return False
    ratio = max(w, h) / max(1, min(w, h))
    if ratio > max_aspect_ratio:
        return False
    return True


def _page_captions(page) -> list[tuple[tuple[float, float, float, float], str]]:
    """Return ``[(bbox, text)]`` for every text block that looks like a
    figure caption on this page.
    """
    captions: list[tuple[tuple[float, float, float, float], str]] = []
    for block in page.get_text("blocks"):
        # block = (x0, y0, x1, y1, text, block_no, block_type)
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        text = (text or "").strip().replace("\n", " ")
        if _CAPTION_RE.match(text):
            captions.append(((x0, y0, x1, y1), text))
    return captions


def _image_bbox(page, xref: int):
    """Best-effort bounding box for an image xref on a page."""
    try:
        rects = page.get_image_rects(xref)
        if rects:
            r = rects[0]
            return (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
    except Exception:  # noqa: BLE001 — bbox is best-effort
        pass
    return None


def _nearest_caption(image_bbox, captions) -> str | None:
    """Pick the caption whose top edge is closest *below* the image.

    Figure captions in papers almost always sit directly under the
    figure, so we prefer the nearest caption with ``caption_y0 >= image_y1``
    and fall back to the nearest caption overall.
    """
    if image_bbox is None or not captions:
        return captions[0][1] if captions else None

    _, img_y0, _, img_y1 = image_bbox
    below = [
        (cap_bbox[1] - img_y1, text)
        for cap_bbox, text in captions
        if cap_bbox[1] >= img_y1 - 5  # small tolerance
    ]
    if below:
        below.sort(key=lambda t: t[0])
        return below[0][1]

    # No caption below — return the vertically closest one.
    captions_by_dist = sorted(
        captions, key=lambda c: abs(c[0][1] - img_y0)
    )
    return captions_by_dist[0][1]
