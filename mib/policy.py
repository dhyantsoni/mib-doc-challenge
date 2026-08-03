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

FEATURE_ORDER = (
    "pages", "ocr_pages", "ocr_conf", "unknown_pages", "hidden_pages", "conflicts",
    "missing_fields", "saw_slip", "has_intake", "has_registry", "has_sponsor", "has_fee",
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
            "sponsor_revoked": sponsor in vocab.MANUAL_REVOKED_SPONSORS,
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
    """Clauses the field manual states without exception, verified on train."""
    flags = set() if record["risk_flags"] == "none" else set(record["risk_flags"].split("|"))
    if flags & set(vocab.DISQUALIFYING_FLAGS):
        return "DENIED"
    if record["visa_class"] == "TRANSIT-7":
        return "DENIED"
    if record["fee_status"] == "unpaid" and not features.get("waiver"):
        return "DENIED"
    if record["fee_status"] == "unknown":
        return "NEEDS_REVIEW"
    if record["arrival_date"] == vocab.UNKNOWN_DATE:
        return "NEEDS_REVIEW"
    return None


def _rule_probabilities(record: dict, features: dict) -> dict[str, float]:
    flags = set() if record["risk_flags"] == "none" else set(record["risk_flags"].split("|"))
    if record["sponsor_id"] in vocab.MANUAL_REVOKED_SPONSORS and record["visa_class"] != "DIP-1":
        return {"APPROVED": 0.05, "DENIED": 0.85, "NEEDS_REVIEW": 0.10}
    if record["sponsor_id"] == vocab.UNKNOWN_SPONSOR and record["visa_class"] != "DIP-1":
        return {"APPROVED": 0.10, "DENIED": 0.35, "NEEDS_REVIEW": 0.55}
    if flags:
        return {"APPROVED": 0.08, "DENIED": 0.25, "NEEDS_REVIEW": 0.67}
    if features.get("conflicts") or features.get("unknown_pages"):
        return {"APPROVED": 0.30, "DENIED": 0.20, "NEEDS_REVIEW": 0.50}
    return {"APPROVED": 0.72, "DENIED": 0.18, "NEEDS_REVIEW": 0.10}


def _load():
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
