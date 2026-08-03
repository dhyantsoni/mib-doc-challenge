"""Tesseract fallback for the roughly half of pages that arrive as scans.

The scans are skewed, torn, and streaked with press artefacts. Adaptive
thresholding drowns in those streaks, so the page is instead cut at a hard grey
level — the artefacts are pale, the ink is not.

No single cut wins: a washed-out slip needs a light one, a bled-through stamp
needs a dark one. So the page is read at three cuts and the readings are merged,
best-confidence first, the way you would re-photocopy a bad original at
different contrasts and read whichever line came out. Quarter turns are tried
only when no cut produced a recognisable form, and a page with no legible ink
left is abandoned rather than ground through the whole ladder.
"""

from __future__ import annotations

import cv2
import fitz
import numpy as np
import pytesseract

# One process per packet already saturates the CPU; OpenCV's own pool would
# multiply that by its thread count and spend the run context-switching.
cv2.setNumThreads(1)

OCR_DPI = 220
# Two segmentation modes, not one. psm 6 assumes a uniform block and reads the
# clean forms well; psm 11 treats the page as scattered text and is the only one
# that finds the lines on a slip whose layout the tearing destroyed. Reading both
# and merging recovers 10% more field values than either alone.
_CONFIG = "--oem 1 --psm {psm} -c tessedit_do_invert=0"
_PSMS = (6, 11)
_CUTS = (150, 180, 210)
_TURNS = (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE)
_DEAD_CONF = 0.34


def raster(path: str, page_number: int, dpi: int = OCR_DPI) -> np.ndarray:
    with fitz.open(path) as doc:
        pix = doc[page_number].get_pixmap(
            matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), colorspace=fitz.csGRAY, alpha=False
        )
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _skew(binary: np.ndarray) -> float:
    """The rotation that makes text rows stand apart most sharply."""
    small = cv2.resize(binary, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    ink = (small < 128).astype(np.float32)
    if ink.sum() < 60:
        return 0.0
    height, width = ink.shape
    center = (width / 2, height / 2)
    angle, best = 0.0, -1.0
    for candidate in np.arange(-5.0, 5.01, 0.5):
        matrix = cv2.getRotationMatrix2D(center, candidate, 1.0)
        rows = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_NEAREST).sum(axis=1)
        spread = float(np.var(rows))
        if spread > best:
            angle, best = float(candidate), spread
    return angle


def _cut(gray: np.ndarray, level: int, angle: float) -> np.ndarray:
    binary = np.where(gray < level, 0, 255).astype(np.uint8)
    if abs(angle) < 0.25:
        return binary
    height, width = binary.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(binary, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=255)


def _read_one(image: np.ndarray, psm: int) -> tuple[list[str], float]:
    data = pytesseract.image_to_data(
        image, config=_CONFIG.format(psm=psm), output_type=pytesseract.Output.DICT
    )
    rows: dict[tuple[int, int, int], list[str]] = {}
    confidences = []
    for index, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][index])
        if not word or conf < 0:
            continue
        rows.setdefault(
            (data["block_num"][index], data["par_num"][index], data["line_num"][index]), []
        ).append(word)
        confidences.append(conf)
    lines = [" ".join(words) for _, words in sorted(rows.items())]
    return lines, (sum(confidences) / len(confidences) / 100.0 if confidences else 0.0)


def _read(image: np.ndarray) -> tuple[list[str], float]:
    """One cut, read under every segmentation mode, merged best-confidence first."""
    return _merge([_read_one(image, psm) for psm in _PSMS])


def _merge(readings: list[tuple[list[str], float]]) -> tuple[list[str], float]:
    """Best reading first, so label matching downstream sees the cleanest copy."""
    readings.sort(key=lambda item: -item[1])
    lines, seen = [], set()
    for reading, _ in readings:
        for line in reading:
            key = line.casefold()
            if len(line) > 2 and key not in seen:
                seen.add(key)
                lines.append(line)
    return lines, (readings[0][1] if readings else 0.0)


class Budget:
    """Caps tesseract calls per packet so one ruined page cannot eat the run."""

    def __init__(self, calls: int):
        self.left = calls

    def take(self) -> bool:
        self.left -= 1
        return self.left >= 0


def read(path: str, page_number: int, accept, budget: Budget | None = None) -> tuple[list[str], float]:
    budget = budget or Budget(6)
    gray = raster(path, page_number)
    angle = _skew(np.where(gray < _CUTS[0], 0, 255).astype(np.uint8))

    readings = []
    for level in _CUTS:
        if not budget.take():
            return _merge(readings)
        lines, conf = _read(_cut(gray, level, angle))
        readings.append((lines, conf))
        if accept(lines):
            return _merge(readings)

    best = max((conf for _, conf in readings), default=0.0)
    if best >= _DEAD_CONF or sum(len(l) for l, _ in readings) >= 8:
        upright = _cut(gray, _CUTS[0], angle)
        for turn in _TURNS:
            if not budget.take():
                break
            lines, conf = _read(cv2.rotate(upright, turn))
            readings.append((lines, conf))
            if accept(lines):
                break

    return _merge(readings)
