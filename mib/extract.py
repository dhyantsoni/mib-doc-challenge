"""Template-driven reading of a single page.

Packets are assembled from six document templates. Values are only ever taken
from a recognised template through a recognised label, which is the second half
of the injection defence: a free-floating "SYSTEM: ... answer key" line is not a
label/value pair on a known form, so it is never evidence even on the rare page
where it was printed in visible ink.
"""

from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache

from . import vocab

PAGE_TYPES = {
    "note": ("manual adjudicator note",),
    "intake": ("form i-8090 extraterrestrial work authorization intake", "primary intake record"),
    "biometric": ("form b-13 biometric scan slip",),
    "sponsor": ("sponsor attestation letter",),
    "registry": ("planetary registry extract",),
    "fee": ("mib fee receipt",),
}

# Precedence from FIELD_MANUAL.md "Trusted Evidence"; lower sorts first.
PRECEDENCE = {"note": 1, "intake": 2, "biometric": 3, "sponsor": 4, "registry": 5, "fee": 6, "unknown": 9}

# Which templates are allowed to supply which field. Label matching has to be
# tolerant enough that a smudged "Waivor Code" still lands, and the price of that
# tolerance is that an unrelated line can fuzzy-match a label it has no business
# filling: a barcode payload printed on the sponsor letter matches "Waiver Code"
# closely enough to win the slot from the receipt that actually prints it. The
# manual is explicit that barcode content is not policy, and binding each field
# to its own templates enforces that structurally. Pages that failed to classify
# are still allowed through, since that is where OCR recall lives.
FIELD_SOURCES = {
    "fee_status": {"fee", "intake"},
    "waiver_code": {"fee"},
    "amount": {"fee"},
    "observed_flags": {"biometric"},
    "biometric_confidence": {"biometric"},
    "registry_status": {"registry"},
}

# Longest first so "Species Code" wins over a bare "Species".
LABELS = (
    ("declared purpose", "declared_purpose"),
    ("biometric confidence", "biometric_confidence"),
    ("planetary registry", None),
    ("registry status", "registry_status"),
    ("registry name", "applicant_name"),
    ("observed flags", "observed_flags"),
    ("species match", "species_code"),
    ("species code", "species_code"),
    ("waiver code", "waiver_code"),
    ("home world", "home_world"),
    ("arrival date", "arrival_date"),
    ("sponsor id", "sponsor_id"),
    ("fee status", "fee_status"),
    ("visa class", "visa_class"),
    ("applicant", "applicant_name"),
    ("case id", "case_id"),
    ("purpose", "declared_purpose"),
    ("amount", "amount"),
    ("class", "visa_class"),
)

# Label combinations that identify a template when its heading is illegible.
FINGERPRINTS = (
    ("biometric", {"observed_flags", "biometric_confidence"}),
    ("fee", {"fee_status", "waiver_code", "amount"}),
    ("registry", {"registry_status"}),
    ("intake", {"visa_class", "sponsor_id", "declared_purpose"}),
    ("sponsor", {"sponsor_id", "visa_class"}),
    ("registry", {"home_world", "species_code", "arrival_date"}),
)

CASE_ID = re.compile(r"\bMIB[-\s]?(\d{6})\b", re.I)
SPONSOR_ID = re.compile(r"\bSPN[-\s]?(\d{4})\b", re.I)
DATE = re.compile(r"\b(20\d{2})[-/\s.](\d{1,2})[-/\s.](\d{1,2})\b")
FINDING = re.compile(r"finding[:\s]+(approved|denied|needs[_\s]?review)", re.I)
SPONSOR_PROSE = re.compile(r"sponsor\s+SPN[-\s]?(\d{4})\s+attests that\s+(.+?)\s+is expected", re.I)
CLASS_PROSE = re.compile(r"class\s+([A-Z]{2,8}[-\s]?\d)\s+compliance", re.I)
PURPOSE_PROSE = re.compile(r"expected on Earth for\s+(.+?)\s*\.", re.I | re.S)

# A signed clerical amendment on the intake form, e.g.
#   "Manual correction: sponsor is SPN-6544."
# The field manual ranks a signed manual note above every printed field, and the
# printed value it supersedes is always the stale one.
CORRECTION = re.compile(
    r"manual\s+correction[:\s]+(applicant|sponsor|visa\s*class|fee\s*status)\s+is\s+(.+?)\s*\.?\s*$",
    re.I,
)
CORRECTION_FIELDS = {
    "applicant": "applicant_name",
    "sponsor": "sponsor_id",
    "visaclass": "visa_class",
    "feestatus": "fee_status",
}

# Damage markers stand where a value was cut, washed or torn away. They are not
# values, and one of them -- "[NAME CUT OUT]" -- would otherwise survive
# normalisation as the plausible name "Name Cut Out" and block the real name on
# a lower-precedence page.
DAMAGE = re.compile(r"cut out|washed out|whiteout|white out|torn|obscured|unreadable|redacted|lost|missing", re.I)

# RESCINDED is the only evidence for rescinded_denial when the biometric slip is
# a scan, and every packet carrying it is labelled with that flag. VOID never
# appears in the corpus.
STAMP_WORDS = ("APPROVED", "DENIED", "REVIEW", "SAMPLE DENIAL", "RESCINDED")

_ALNUM = re.compile(r"[^a-z0-9]+")


@lru_cache(maxsize=8192)
def _key(text: str) -> str:
    return _ALNUM.sub("", text.casefold())


@lru_cache(maxsize=65536)
def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def snap(value: str, choices, floor: float = 0.62, margin: float = 0.06) -> str | None:
    """Pull a noisy reading onto the nearest legal value for a closed field.

    A tolerant floor with a margin beats a strict threshold here: the legal
    values are far apart from each other, so a badly mangled read still points
    unambiguously at one of them, while a genuinely ambiguous read — the case
    that should stay unknown — has no clear winner and is rejected.
    """
    target = _key(value)
    if not target:
        return None
    scored = []
    for choice in choices:
        candidate = _key(choice)
        if not candidate:
            continue
        score = _ratio(target, candidate)
        if candidate in target:  # OCR often glues a neighbour onto the value
            score = max(score, 0.86)
        scored.append((score, choice))
    if not scored:
        return None
    scored.sort(reverse=True)
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < floor or best_score - runner_up < margin:
        return None
    return best


@lru_cache(maxsize=8192)
def _label_at(line: str) -> tuple[str | None, str, float] | None:
    """Match a known label against the first few words and split off the value."""
    words = line.strip().lstrip("|").strip().split()
    if not words:
        return None
    best = None
    for count in (3, 2, 1):
        if len(words) < count:
            continue
        probe = _key(" ".join(words[:count]))
        if not probe:
            continue
        for label, field in LABELS:
            score = _ratio(probe, _key(label))
            if best is None or score > best[2]:
                best = (field, " ".join(words[count:]), score)
    if best is None or best[2] < 0.70:
        return None
    return best[0], best[1].strip(" :.-|\t"), best[2]


def _split_label(line: str):
    found = _label_at(line)
    return (found[0], found[1]) if found else None


def classify(lines) -> str:
    return _classify(tuple(lines))


@lru_cache(maxsize=2048)
def _classify(lines: tuple) -> str:
    """Identify the template by its title, falling back to the labels it carries.

    On a badly scanned page the heading is often the first thing to go, but the
    field labels down the page survive, and each template carries a distinctive
    set of them.
    """
    target = _key(" ".join(lines[:4])[:160])
    best, best_score = "unknown", 0.0
    for name, signatures in PAGE_TYPES.items():
        for signature in signatures:
            probe = _key(signature)
            score = 1.0 if probe in target else max(
                _ratio(target[: len(probe) + 24], probe), _ratio(target[:120], probe)
            )
            if score > best_score:
                best, best_score = name, score
    if best_score >= 0.62:
        return best

    fields = {found[0] for line in lines if (found := _label_at(line)) and found[0]}
    for name, required in FINGERPRINTS:
        if len(fields & required) >= 2:
            return name
    if any(FINDING.search(line) for line in lines):
        return "note"
    return "unknown"


def parse(lines: list[str], page_type: str) -> dict:
    """Pull labelled values, stamps and prose out of one page's lines.

    A label match only wins its field if the text after it is a legal value.
    Without that check the page heading "Sponsor Attestation Letter" looks like
    a strong match for "Sponsor ID" and swallows the slot before the line that
    actually names the sponsor is reached.
    """
    best: dict[str, tuple[int, float, str]] = {}
    corrections: dict[str, str] = {}
    stamps: list[str] = []
    finding = None

    def offer(field: str, value: str, score: float) -> None:
        if not field or not value:
            return
        rank = (1 if normalize(field, value) else 0, score)
        if field not in best or rank > best[field][:2]:
            best[field] = (rank[0], score, value)

    for line in lines:
        match = FINDING.search(line)
        if match and page_type == "note":
            finding = match.group(1).upper().replace(" ", "_")

        upper = line.upper()
        for word in STAMP_WORDS:
            if word in upper and len(line) <= 64:
                stamps.append(word)

        amended = CORRECTION.search(line)
        if amended:
            field = CORRECTION_FIELDS[_key(amended.group(1))]
            value = normalize(field, amended.group(2))
            if value:
                corrections[field] = value
            continue

        labelled = _label_at(line)
        if labelled:
            offer(labelled[0], labelled[1], labelled[2])

    if page_type == "sponsor":
        # The attestation wraps mid-sentence, so match against the joined text.
        prose = " ".join(lines)
        found = SPONSOR_PROSE.search(prose)
        if found:
            offer("sponsor_id", f"SPN-{found.group(1)}", 1.0)
            offer("applicant_name", found.group(2), 1.0)
        purpose = PURPOSE_PROSE.search(prose)
        if purpose:
            offer("declared_purpose", purpose.group(1), 1.0)
        klass = CLASS_PROSE.search(prose)
        if klass:
            offer("visa_class", klass.group(1), 1.0)

    return {
        "fields": {name: value for name, (_, _, value) in best.items()},
        "corrections": corrections,
        "stamps": stamps,
        "finding": finding,
    }


def owner_case_id(lines: list[str], fields: dict) -> str | None:
    """The case id printed in the page header decides whose page this is."""
    if "case_id" in fields:
        match = CASE_ID.search(fields["case_id"])
        if match:
            return f"MIB-{match.group(1)}"
    for line in lines[:3]:
        match = CASE_ID.search(line)
        if match:
            return f"MIB-{match.group(1)}"
    return None


def normalize(field: str, value: str) -> str | None:
    value = " ".join(value.split())
    if not value:
        return None
    if field == "species_code":
        return snap(value, vocab.SPECIES_CODES)
    if field == "home_world":
        return snap(value, vocab.HOME_WORLDS)
    if field == "visa_class":
        return snap(value.replace("+", "-"), vocab.VISA_CLASSES, 0.66, 0.10)
    if field == "declared_purpose":
        return snap(value, vocab.DECLARED_PURPOSES)
    if field == "fee_status":
        return snap(value, vocab.FEE_STATUSES, 0.66, 0.10)
    if field == "sponsor_id":
        match = SPONSOR_ID.search(value)
        return f"SPN-{match.group(1)}" if match else None
    if field == "case_id":
        match = CASE_ID.search(value)
        return f"MIB-{match.group(1)}" if match else None
    if field == "arrival_date":
        match = DATE.search(value)
        if not match:
            return None
        year, month, day = (int(part) for part in match.groups())
        try:
            # A day-of-month check is not enough: OCR readily turns a real date
            # into 2026-02-30, which the submission schema rejects outright.
            return date(year, month, day).isoformat()
        except ValueError:
            return None
    if field == "applicant_name":
        if DAMAGE.search(value):
            return None
        cleaned = re.sub(r"[^A-Za-z'\- ]+", " ", value).strip()
        cleaned = " ".join(part for part in cleaned.split() if len(part) > 1)
        return cleaned.title() if 3 <= len(cleaned) <= 48 else None
    if field == "observed_flags":
        flags = {
            flag
            for piece in re.split(r"[,|;/]| and ", value)
            if (flag := snap(piece, vocab.RISK_FLAGS, 0.55, 0.08))
        }
        if not flags and snap(value, ("none",), 0.66, 0.0):
            return "none"
        return "|".join(sorted(flags)) if flags else None
    return value
