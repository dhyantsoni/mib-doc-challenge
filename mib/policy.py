"""Adjudication: manual rules as constraints, a learned model for the rest.

Two things separate this from a plain classifier. The field manual's unambiguous
clauses are applied as hard constraints the model cannot overrule, so a packet
carrying a disqualifying flag can never be approved. And the emitted decision
maximises expected score under the published payoff matrix rather than picking
the likeliest class — with a false approval costing -4 and a wrong denial
costing 0, those are different answers whenever APPROVED and DENIED are close.
"""

from __future__ import annotations

import os

from . import vocab

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# The manual publishes three revoked sponsors and adds "Other revoked sponsors
# may appear in examples". The packets name the rest outright — an adjudicator
# note reads "Finding: DENIED. Reason: Revoked sponsor: SPN-2718." Each of these
# six is used 13-20 times across training while every other sponsor appears once
# or twice, which is what a policy list looks like from the outside.
REVOKED_SPONSORS = frozenset(vocab.MANUAL_REVOKED_SPONSORS) | {"SPN-2718", "SPN-7331", "SPN-9090"}

# Worlds whose registry extract prints "Registry Status EMBARGO REVIEW"; the
# other ten print "CLEAR". The registry line is the primary signal and the one
# that generalises; this list is the fallback for when that page is missing.
EMBARGOED_WORLDS = frozenset({"Eris Relay", "TRAPPIST-1e", "Wolf-1061c"})

# No packet prints a receipt date, so "180 days before packet receipt" has to be
# anchored to the batch: the latest arrival anywhere in training is 2026-07-12,
# and 180 days earlier lands inside a 49-day gap separating the stale denials
# from everything else, so the exact anchor within that gap does not matter.
STALE_BEFORE = "2026-01-13"

FEATURE_ORDER = (
    "pages", "ocr_pages", "ocr_conf", "unknown_pages", "hidden_pages", "conflicts",
    "missing_fields", "saw_slip", "fee_seen", "date_damaged", "has_intake", "has_registry", "has_sponsor", "has_fee",
    "has_note", "foreign_pages", "stamp_denied", "stamp_approved", "stamp_review",
    "stamp_sample", "waiver", "registry_embargo", "n_flags", "n_disqualifying",
    "n_review_flags", "identity_mismatch", "sponsor_conflict",
    "fee_paid", "fee_waived", "fee_unpaid", "fee_unknown",
    "visa_xw1", "visa_xw2", "visa_dip1", "visa_med3", "visa_transit7", "visa_unknown",
    "sponsor_missing", "sponsor_revoked", "date_missing", "date_age", "name_missing",
    "species_missing", "world_missing", "purpose_missing",
    "flag_memory_tampering", "flag_planetary_embargo", "flag_active_warrant",
    "flag_biohazard_red", "flag_identity_conflict", "flag_sponsor_mismatch",
    "flag_illegible_biometrics", "flag_rescinded_denial",
)

_REFERENCE_DAY = 739_000  # fixed epoch so arrival age is stable across runs
_model = None


def _ordinal(iso: str) -> int | None:
    try:
        year, month, day = (int(part) for part in iso.split("-"))
        return year * 372 + month * 31 + day
    except (ValueError, AttributeError):
        return None


_STALE_ORDINAL = _ordinal(STALE_BEFORE)


def vectorize(record: dict, features: dict) -> list[float]:
    flags = set() if record["risk_flags"] == "none" else set(record["risk_flags"].split("|"))
    visa = record["visa_class"]
    fee = record["fee_status"]
    sponsor = record["sponsor_id"]
    day = _ordinal(record["arrival_date"])

    row = {name: float(features.get(name, 0)) for name in FEATURE_ORDER}
    row.update(
        {
            "fee_paid": fee == "paid",
            "fee_waived": fee == "waived",
            "fee_unpaid": fee == "unpaid",
            "fee_unknown": fee not in vocab.FEE_STATUSES or fee == "unknown",
            "visa_xw1": visa == "XW-1",
            "visa_xw2": visa == "XW-2",
            "visa_dip1": visa == "DIP-1",
            "visa_med3": visa == "MED-3",
            "visa_transit7": visa == "TRANSIT-7",
            "visa_unknown": visa not in vocab.VISA_CLASSES,
            "sponsor_missing": sponsor == vocab.UNKNOWN_SPONSOR,
            "sponsor_revoked": sponsor in REVOKED_SPONSORS,
            "date_missing": record["arrival_date"] == vocab.UNKNOWN_DATE,
            "date_age": 0.0 if day is None else (_REFERENCE_DAY - day) / 372.0,
            "name_missing": record["applicant_name"] == vocab.UNKNOWN_TEXT,
            "species_missing": record["species_code"] == vocab.UNKNOWN_TEXT,
            "world_missing": record["home_world"] == vocab.UNKNOWN_TEXT,
            "purpose_missing": record["declared_purpose"] == vocab.UNKNOWN_TEXT,
        }
    )
    row.update({f"flag_{name}": name in flags for name in vocab.RISK_FLAGS})
    return [float(row[name]) for name in FEATURE_ORDER]


def constrain(record: dict, features: dict) -> str | None:
    """Clauses that hold without exception across the training packets.

    Given the true fields these decide 973 of 1000 cases, and the remaining 27
    are the damaged-date clause at the end. The ordering is load-bearing rather
    than cosmetic: 95 packets carry a review-only flag *and* a revoked sponsor,
    an embargoed world, or a stale arrival date, and every one of them is denied,
    so every denial clause has to be tested before every review clause.
    """
    flags = set() if record["risk_flags"] == "none" else set(record["risk_flags"].split("|"))
    visa = record["visa_class"]

    if flags & set(vocab.DISQUALIFYING_FLAGS):
        return "DENIED"  # 186/186
    if visa == "TRANSIT-7":
        return "DENIED"  # 53/53
    if record["fee_status"] == "unpaid":
        return "DENIED"  # 50/50, including every unpaid DIP-1 packet
    # Only a receipt that was read and says `unknown` is the manual's gap; a
    # receipt this pipeline could not find is a gap in the reading, not the packet.
    if record["fee_status"] == "unknown" and features.get("fee_seen"):
        return "NEEDS_REVIEW"  # 44/44

    if visa != "DIP-1":
        # The manual exempts DIP-1 from the sponsor requirement and from
        # staleness; the labels extend the same exemption to the embargo clause.
        # An embargoed world still denies a DIP-1 packet when the biometric slip
        # actually observed planetary_embargo, which is the first clause above.
        if record["sponsor_id"] in REVOKED_SPONSORS:
            return "DENIED"  # 79/79
        if features.get("registry_embargo") or record["home_world"] in EMBARGOED_WORLDS:
            return "DENIED"  # 31/31 here, 88/88 unconditionally
        day = _ordinal(record["arrival_date"])
        if record["arrival_date"] != vocab.UNKNOWN_DATE and day is not None and day < _STALE_ORDINAL:
            return "DENIED"  # 32/32

    if flags & set(vocab.REVIEW_FLAGS):
        return "NEEDS_REVIEW"  # 209/209
    if features.get("date_damaged"):
        return "NEEDS_REVIEW"  # 27/27 -- the slot is present but destroyed
    return None


def _rule_probabilities(record: dict, features: dict) -> dict[str, float]:
    """Prior for the packets no constraint matched.

    Every training packet in that residue is approved, so the base case is
    confident. What is left to model there is legibility, not policy.
    """
    flags = set() if record["risk_flags"] == "none" else set(record["risk_flags"].split("|"))
    if record["sponsor_id"] == vocab.UNKNOWN_SPONSOR and record["visa_class"] != "DIP-1":
        # Every training packet names a sponsor, so a blank one is a reading
        # failure rather than a disqualifying condition.
        return {"APPROVED": 0.12, "DENIED": 0.10, "NEEDS_REVIEW": 0.78}
    if flags:
        return {"APPROVED": 0.05, "DENIED": 0.05, "NEEDS_REVIEW": 0.90}
    if features.get("conflicts") or features.get("unknown_pages"):
        return {"APPROVED": 0.35, "DENIED": 0.07, "NEEDS_REVIEW": 0.58}
    return {"APPROVED": 0.88, "DENIED": 0.04, "NEEDS_REVIEW": 0.08}


def _load():
    """Load the model artifact that ships inside the image.

    This is our own build output, baked in at image build time by tools/train.py
    and never read from the input mount, so it is not an untrusted deserialise.
    """
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        import joblib

        _model = joblib.load(MODEL_PATH)
    return _model


def probabilities(record: dict, features: dict) -> dict[str, float]:
    finding = features.get("finding")
    if finding in vocab.ADJUDICATIONS:
        # A signed adjudicator note is the top of the evidence order and has
        # never disagreed with the label on the training packets.
        return {name: (0.97 if name == finding else 0.015) for name in vocab.ADJUDICATIONS}

    model = _load()
    if model is None:
        return _rule_probabilities(record, features)

    row = [vectorize(record, features)]
    scores = model["classifier"].predict_proba(row)[0]
    return {name: float(score) for name, score in zip(model["classifier"].classes_, scores)}


def decide(record: dict, features: dict) -> tuple[str, float]:
    probability = probabilities(record, features)
    forced = constrain(record, features)
    if forced:
        probability = {
            name: 0.90 * (name == forced) + 0.10 * probability.get(name, 0.0)
            for name in vocab.ADJUDICATIONS
        }

    total = sum(probability.values()) or 1.0
    probability = {name: value / total for name, value in probability.items()}

    action = max(
        vocab.ADJUDICATIONS,
        key=lambda choice: sum(vocab.PAYOFF[choice][truth] * probability[truth] for truth in vocab.ADJUDICATIONS),
    )

    model = _load()
    confidence = probability[action]
    if model is not None and model.get("calibrator") is not None:
        confidence = float(model["calibrator"].predict([confidence])[0])
    return action, round(min(0.99, max(0.01, confidence)), 4)
