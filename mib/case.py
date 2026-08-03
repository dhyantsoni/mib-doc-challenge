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

# The arrival-date slot holds either a date or a damage marker; the corpus uses
# "UNREADABLE" and "[DATE WASHED OUT]", both digit-free. Testing for the absence
# of digits catches both without firing on a garbled read of a real date, which
# should stay a reading problem rather than become a review decision.
_DIGIT = re.compile(r"\d")


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
    corrections: dict[str, str] = field(default_factory=dict)
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
        corrections=parsed["corrections"],
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
            allowed = extract.FIELD_SOURCES.get(raw_field)
            if allowed and sheet.kind != "unknown" and sheet.kind not in allowed:
                continue
            value = extract.normalize(raw_field, raw_value)
            if value:
                candidates.setdefault(raw_field, []).append((rank, sheet.kind, value))
        # A signed amendment outranks every printed field, per the manual's
        # "visible MIB adjudicator stamp or signed manual note".
        for raw_field, value in sheet.corrections.items():
            candidates.setdefault(raw_field, []).append((0, "correction", value))

    resolved, conflicts = {}, {}
    amended = {name for sheet in sheets for name in sheet.corrections}
    for name, found in candidates.items():
        found.sort(key=lambda item: item[0])
        if name == "applicant_name":
            # The only open-vocabulary field, so it is the only one a closed
            # vocabulary cannot repair, and the only one where a single bad
            # reading survives normalisation. Where several pages name the
            # applicant, agreement between them beats the precedence order --
            # two independent pages rarely invent the same misreading.
            tally: dict[str, list] = {}
            for rank, kind, value in found:
                tally.setdefault(value, []).append(rank)
            resolved[name] = min(tally, key=lambda v: (-len(tally[v]), min(tally[v])))
        else:
            resolved[name] = found[0][2]
        top = found[0][0]
        conflicts[name] = len({value for rank, _, value in found if rank == top}) - 1
        if not conflicts[name] and len({value for _, _, value in found}) > 1:
            conflicts[name] = 1
        if name in amended:
            # A disagreement the paperwork itself explains is not a conflict.
            conflicts[name] = 0
    return resolved, conflicts


def _named_by(sheets: list[Sheet], case_id: str) -> dict[str, str]:
    """The applicant name each template gives, for cross-page identity checks.

    Only clean readings count. Two OCR'd pages will disagree about a name from
    scan noise alone, and treating that as an identity conflict invents far more
    flags than it recovers -- names are the one open-vocabulary field, so there
    is no legal-value check to catch the difference.
    """
    names: dict[str, str] = {}
    for sheet in sheets:
        if sheet.ocr or (sheet.owner and sheet.owner != case_id):
            continue
        value = extract.normalize("applicant_name", sheet.values.get("applicant_name", ""))
        if value:
            names.setdefault(sheet.kind, value)
    return names


def _risk_flags(sheets: list[Sheet], case_id: str) -> tuple[str, bool]:
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

    # A legible slip states the flags outright and is complete on its own. Only
    # when it is unreadable are the flags reconstructed from what the rest of the
    # packet betrays: two records naming different applicants with no clerical
    # amendment to explain it is an identity conflict, whereas a sponsor letter
    # attesting for someone else is a sponsor mismatch. Distinguishing the two by
    # which template disagrees is the whole of the difference.
    if not seen:
        names = _named_by(sheets, case_id)
        amended = {name for sheet in sheets for name in sheet.corrections}
        records = {kind: value for kind, value in names.items() if kind in ("intake", "registry", "biometric")}
        if len(set(records.values())) > 1 and "applicant_name" not in amended:
            flags.add("identity_conflict")
        if records and names.get("sponsor") and names["sponsor"] not in set(records.values()):
            flags.add("sponsor_mismatch")
        if any("RESCINDED" in sheet.stamps for sheet in sheets):
            flags.add("rescinded_denial")

    # An EMBARGO REVIEW registry status is a denial ground, not an observed risk
    # flag: most embargoed-world packets print it while their biometric slip
    # observes no embargo, and the labels follow the slip. policy.constrain reads
    # the registry_embargo feature instead of corrupting the flags with it.
    return ("|".join(sorted(flags)) if flags else "none"), seen


def _fee_status(resolved: dict) -> tuple[str, bool]:
    """Read the fee, and say whether a receipt was actually read at all.

    The receipt prints an amount and a waiver code beside the status, and those
    survive damage that takes the status line out: a diplomatic waiver code
    always meant `waived` on the training packets, and a non-zero amount always
    meant `paid`.

    The distinction that matters is between a receipt that says `unknown` — a
    genuine gap in the packet, which the manual sends to review — and a receipt
    this pipeline simply failed to find. Reporting the second as `unknown` both
    loses the field and drags an otherwise decidable case into review, so an
    unread receipt falls back to the overwhelmingly common `paid` and is marked
    as unseen for the model to discount.
    """
    printed = resolved.get("fee_status")
    waiver = "WAIVER" in resolved.get("waiver_code", "").upper()
    amount = re.search(r"(\d+(?:\.\d+)?)", resolved.get("amount", "").replace(",", ""))

    # The printed word is the slot that gets corrupted; the amount and the waiver
    # code beside it never disagree with the truth. A non-zero amount always
    # meant paid and a waiver code always meant waived, so those are read first
    # and the word is only consulted once both are silent.
    if amount and float(amount.group(1)) > 0:
        return "paid", True
    if waiver:
        return "waived", True
    if amount:
        return ("unpaid" if printed == "unpaid" else "unknown"), True

    if printed in vocab.FEE_STATUSES:
        return printed, True
    return "paid", False


def _features(sheets: list[Sheet], resolved: dict, conflicts: dict, flags: str, saw_slip: bool, fee_seen: bool) -> dict:
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
        "date_damaged": int(
            any(
                (not sheet.owner or sheet.owner == resolved.get("case_id", ""))
                and not _DIGIT.search(sheet.values.get("arrival_date", "0"))
                for sheet in sheets
            )
        ),
        "missing_fields": sum(1 for name in OUTPUT_FIELDS if name not in resolved and name != "risk_flags"),
        "saw_slip": int(saw_slip),
        "fee_seen": int(fee_seen),
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
        "waiver": int("WAIVER" in resolved.get("waiver_code", "").upper()),
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
    flags, saw_slip = _risk_flags(sheets, case_id)
    fee, fee_seen = _fee_status(resolved)

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
        "fee_status": fee,
    }
    return {
        "record": record,
        "features": _features(sheets, resolved, conflicts, flags, saw_slip, fee_seen),
        "resolved": resolved,
    }
