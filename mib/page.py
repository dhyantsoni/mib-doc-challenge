"""Page decoding with an ink-verification gate.

The packets hide text in several ways: white glyphs on white paper, spans placed
outside the page crop, invisible render modes, and text buried under an image.
Rather than a rule per trick, every span from the PDF text layer is checked
against the rendered raster: a span the eye cannot see leaves no ink where it
claims to be. That single test subsumes all of them, and it is what separates
trusted evidence from prompt injection everywhere downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz
import numpy as np

RASTER_DPI = 110
_ZOOM = RASTER_DPI / 72.0

# A span is visible if enough of its box departs from the local paper tone.
_INK_DELTA = 28  # grey levels away from background that counts as ink
_MIN_INK_RATIO = 0.012  # fraction of the box that must carry ink
_MIN_COLOR_CONTRAST = 20  # declared fill vs. paper, in grey levels

_WS = re.compile(r"\s+")


@dataclass
class Span:
    text: str
    bbox: tuple[float, float, float, float]
    page: int
    size: float = 0.0
    color: int = 0
    visible: bool = True
    ink: float = 0.0
    conf: float = 1.0
    ocr: bool = False

    @property
    def y(self) -> float:
        return self.bbox[1]

    @property
    def x(self) -> float:
        return self.bbox[0]


@dataclass
class Page:
    number: int
    width: float
    height: float
    spans: list[Span] = field(default_factory=list)
    gray: np.ndarray | None = None
    rotation: int = 0

    @property
    def visible_spans(self) -> list[Span]:
        return [s for s in self.spans if s.visible]

    @property
    def hidden_spans(self) -> list[Span]:
        return [s for s in self.spans if not s.visible]


def _luminance(color: int) -> float:
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return 0.299 * r + 0.587 * g + 0.114 * b


def _raster(page: fitz.Page) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(_ZOOM, _ZOOM), colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _verify(span: Span, gray: np.ndarray, origin: tuple[float, float]) -> None:
    """Set span.visible/span.ink by looking at the pixels the span covers."""
    h, w = gray.shape
    x0 = int((span.bbox[0] - origin[0]) * _ZOOM)
    y0 = int((span.bbox[1] - origin[1]) * _ZOOM)
    x1 = int(round((span.bbox[2] - origin[0]) * _ZOOM))
    y1 = int(round((span.bbox[3] - origin[1]) * _ZOOM))

    # Clipping to the raster is the off-crop test: a span pushed past the page
    # edge keeps no visible area at all.
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))
    box_area = max(1, (x1 - x0) * (y1 - y0))
    if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < 0.5 * box_area:
        span.visible = False
        return

    patch = gray[cy0:cy1, cx0:cx1]
    # Paper tone from a ring around the span, falling back to the box itself.
    ring = gray[max(0, cy0 - 4):cy1 + 4, max(0, cx0 - 6):cx1 + 6]
    paper = float(np.median(ring)) if ring.size else float(np.median(patch))

    span.ink = float(np.count_nonzero(np.abs(patch.astype(np.int16) - paper) > _INK_DELTA)) / patch.size
    declared_contrast = abs(_luminance(span.color) - paper)
    span.visible = span.ink >= _MIN_INK_RATIO and declared_contrast >= _MIN_COLOR_CONTRAST


def load(path: str, render: bool = True) -> list[Page]:
    pages: list[Page] = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc):
            rect = page.rect
            out = Page(number=index, width=rect.width, height=rect.height, rotation=page.rotation)
            if render:
                out.gray = _raster(page)
            raw = page.get_text("dict")
            for block in raw.get("blocks", ()):
                for line in block.get("lines", ()):
                    for span in line.get("spans", ()):
                        text = _WS.sub(" ", span.get("text", "")).strip()
                        if not text:
                            continue
                        item = Span(
                            text=text,
                            bbox=tuple(span["bbox"]),
                            page=index,
                            size=float(span.get("size", 0.0)),
                            color=int(span.get("color", 0)),
                        )
                        if out.gray is not None:
                            _verify(item, out.gray, (rect.x0, rect.y0))
                        out.spans.append(item)
            out.spans.sort(key=lambda s: (round(s.y, 1), s.x))
            pages.append(out)
    return pages


def lines_of(spans: list[Span], tolerance: float = 3.0) -> list[tuple[float, list[Span]]]:
    """Group spans into visual lines so label/value pairs survive span splitting."""
    grouped: list[tuple[float, list[Span]]] = []
    for span in sorted(spans, key=lambda s: (round(s.y, 1), s.x)):
        if grouped and abs(span.y - grouped[-1][0]) <= tolerance:
            grouped[-1][1].append(span)
        else:
            grouped.append((span.y, [span]))
    return [(y, sorted(items, key=lambda s: s.x)) for y, items in grouped]


def text_of(spans: list[Span]) -> str:
    return "\n".join(" ".join(s.text for s in items) for _, items in lines_of(spans))
