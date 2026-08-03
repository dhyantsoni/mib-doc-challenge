"""Assemble one packet into a record plus the evidence features behind it.

Every value keeps the page it came from, so conflicts are resolved by the field
manual's precedence order rather than by whichever page happened to be read
last, and disagreements survive as features instead of being averaged away.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from . import extract, ocr, page as pagelib, vocab

OUTPUT_FIELDS = (
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
)

_MIN_VISIBLE_SPANS = 5


@dataclass
class Sheet:
    index: int
    kind: str
    owner: str | None
    values: dict[str, str]
    stamps: list[str]
    finding: str | None
    ocr: bool
    confidence: float
    hidden_values: dict[str, str] = field(default_factory=dict)


def _lines(spans) -> list[str]:
    return [" ".join(s.text for s in items) for _, items in pagelib.lines_of(spans)]


def _accepted(lines: list[str]) -> bool:
    kind = extract.classify(lines)
    if kind == "unknown":
        return False
    if kind == "note":
        return any(extract.FINDING.search(line) for line in lines)
    return sum(1 for line in lines if extract._split_label(line)) >= 2


def _sheet(path: str, page, index: int, allow_ocr: bool, budget) -> Sheet:
    visible = _lines(page.visible_spans)
    kind = extract.classify(visible)
    used_ocr = False
    confidence = 1.0

    if allow_ocr and (kind == "unknown" or len(page.visible_spans) < _MIN_VISIBLE_SPANS):
        recovered, confidence = ocr.read(path, index, _accepted, budget)
        if extract.classify(recovered) != "unknown":
            visible, kind, used_ocr = recovered, extract.classify(recovered), True
        elif recovered:
            visible, used_ocr = visible + recovered, True

    parsed = extract.parse(visible, kind)
    hidden = extract.parse(_lines(page.hidden_spans), kind)["fields"] if page.hidden_spans else {}

    return Sheet(
        index=index,
        kind=kind,
        owner=extract.owner_case_id(visible, parsed["fields"]),
        values=parsed["fields"],
        stamps=parsed["stamps"],
        finding=parsed["finding"],
        ocr=used_ocr,
        confidence=confidence,
        hidden_values=hidden,
    )


def _resolve(sheets: list[Sheet], case_id: str) -> tuple[dict, dict]:
    """Highest-precedence trusted reading per field, plus conflict counts."""
    candidates: dict[str, list[tuple[int, str, str]]] = {}
    for sheet in sheets:
        if sheet.owner and sheet.owner != case_id:
            continue  # a page belonging to another applicant in the same packet
        rank = extract.PRECEDENCE.get(sheet.kind, 9)
        for raw_field, raw_value in sheet.values.items():
            value = extract.normalize(raw_field, raw_value)
            if value:
                candidates.setdefault(raw_field, []).append((rank, sheet.kind, value))

    resolved, conflicts = {}, {}
    for name, found in candidates.items():
        found.sort(key=lambda item: item[0])
        resolved[name] = found[0][2]
        top = found[0][0]
        conflicts[name] = len({value for rank, _, value in found if rank == top}) - 1
        if not conflicts[name] and len({value for _, _, value in found}) > 1:
            conflicts[name] = 1
    return resolved, conflicts


def _risk_flags(sheets: list[Sheet], resolved: dict, case_id: str) -> tuple[str, bool]:
    observed = None
    for sheet in sorted(sheets, key=lambda s: extract.PRECEDENCE.get(s.kind, 9)):
        if sheet.owner and sheet.owner != case_id:
            continue
        if "observed_flags" in sheet.values:
            value = extract.normalize("observed_flags", sheet.values["observed_flags"])
            if value:
                observed = value
                break

    flags = set() if observed in (None, "none") else set(observed.split("|"))
    seen = observed is not None

    if any("SAMPLE DENIAL" not in s.stamps and "RESCINDED" in s.stamps for s in sheets):
        flags.add("rescinded_denial")
    if resolved.get("registry_status", "").upper().startswith("EMBARGO"):
        flags.add("planetary_embargo")

    return ("|".join(sorted(flags)) if flags else "none"), seen


def _features(sheets: list[Sheet], resolved: dict, conflicts: dict, flags: str, saw_slip: bool) -> dict:
    kinds = {sheet.kind for sheet in sheets}
    flag_set = set() if flags == "none" else set(flags.split("|"))
    finding = next((s.finding for s in sorted(sheets, key=lambda s: s.index) if s.finding), None)
    stamps = [word for sheet in sheets for word in sheet.stamps]

    return {
        "finding": finding or "",
        "pages": len(sheets),
        "ocr_pages": sum(1 for s in sheets if s.ocr),
        "ocr_conf": min((s.confidence for s in sheets if s.ocr), default=1.0),
        "unknown_pages": sum(1 for s in sheets if s.kind == "unknown"),
        "hidden_pages": sum(1 for s in sheets if s.hidden_values),
        "conflicts": sum(conflicts.get(name, 0) for name in OUTPUT_FIELDS),
        "missing_fields": sum(1 for name in OUTPUT_FIELDS if name not in resolved and name != "risk_flags"),
        "saw_slip": int(saw_slip),
        "has_intake": int("intake" in kinds),
        "has_registry": int("registry" in kinds),
        "has_sponsor": int("sponsor" in kinds),
        "has_fee": int("fee" in kinds),
        "has_note": int("note" in kinds),
        "foreign_pages": sum(1 for s in sheets if s.owner and s.owner != resolved.get("case_id", "")),
        "stamp_denied": int("DENIED" in stamps),
        "stamp_approved": int("APPROVED" in stamps),
        "stamp_review": int("REVIEW" in stamps),
        "stamp_sample": int("SAMPLE DENIAL" in stamps),
        "waiver": int(bool(resolved.get("waiver_code", "N/A").strip().upper() not in {"N/A", "NA", "NONE", ""})),
        "registry_embargo": int(resolved.get("registry_status", "").upper().startswith("EMBARGO")),
        "n_flags": len(flag_set),
        "n_disqualifying": len(flag_set & set(vocab.DISQUALIFYING_FLAGS)),
        "n_review_flags": len(flag_set & set(vocab.REVIEW_FLAGS)),
        "identity_mismatch": int(conflicts.get("applicant_name", 0) > 0),
        "sponsor_conflict": int(conflicts.get("sponsor_id", 0) > 0),
    }


def read_packet(path: str, allow_ocr: bool = True) -> dict:
    case_id = os.path.basename(path)[:-4]
    if not re.fullmatch(r"MIB-\d{6}", case_id):
        case_id = ""

    pages = pagelib.load(path)
    budget = ocr.Budget(4 + 2 * sum(1 for p in pages if len(p.visible_spans) < _MIN_VISIBLE_SPANS))
    sheets = [_sheet(path, page, index, allow_ocr, budget) for index, page in enumerate(pages)]

    resolved, conflicts = _resolve(sheets, case_id)
    resolved.setdefault("case_id", case_id)
    flags, saw_slip = _risk_flags(sheets, resolved, case_id)

    record = {
        "case_id": case_id or resolved.get("case_id", ""),
        "applicant_name": resolved.get("applicant_name", vocab.UNKNOWN_TEXT),
        "species_code": resolved.get("species_code", vocab.UNKNOWN_TEXT),
        "home_world": resolved.get("home_world", vocab.UNKNOWN_TEXT),
        "visa_class": resolved.get("visa_class", vocab.UNKNOWN_TEXT),
        "sponsor_id": resolved.get("sponsor_id", vocab.UNKNOWN_SPONSOR),
        "arrival_date": resolved.get("arrival_date", vocab.UNKNOWN_DATE),
        "declared_purpose": resolved.get("declared_purpose", vocab.UNKNOWN_TEXT),
        "risk_flags": flags,
        "fee_status": resolved.get("fee_status", "unknown"),
    }
    return {
        "record": record,
        "features": _features(sheets, resolved, conflicts, flags, saw_slip),
        "resolved": resolved,
    }
