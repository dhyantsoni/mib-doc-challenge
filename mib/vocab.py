"""Closed vocabularies and adjudication constants from FIELD_MANUAL.md.

Every categorical field in the packet schema draws from a small fixed set. Snapping
noisy OCR output onto these sets is what stops "ORION_GRAYS" from being read as
"0RION_CRAYS", and it is why a wrong character is usually recoverable.
"""

SPECIES_CODES = (
    "ALPHA_DRACONIAN",
    "ANDROMEDAN",
    "AQUARIAN_MANTIS",
    "ARCTURIAN",
    "CENTAURI_SYNTH",
    "JOVIAN_GASFORM",
    "KAIJU_MICRO",
    "LUNA_SECURID",
    "ORION_GRAYS",
    "SIRIUS_AVIAN",
    "TRIANGULAN",
    "VENUSIAN_MYCELIAL",
)

HOME_WORLDS = (
    "Barnard-c",
    "Eris Relay",
    "Europa Station",
    "Gliese-581g",
    "Kepler-186f",
    "Luyten-b",
    "Mars Dome-7",
    "Proxima-b",
    "Sirius Outpost",
    "TRAPPIST-1e",
    "Titan Freeport",
    "Wolf-1061c",
    "Zeta Reticuli",
)

VISA_CLASSES = ("DIP-1", "MED-3", "TRANSIT-7", "XW-1", "XW-2")

DECLARED_PURPOSES = (
    "archive audit",
    "cultural exchange",
    "diplomatic",
    "field repair",
    "medical consult",
    "reactor maintenance",
    "research",
    "transit",
    "translation",
    "xenobotany",
)

FEE_STATUSES = ("paid", "waived", "unpaid", "unknown")

DISQUALIFYING_FLAGS = (
    "active_warrant",
    "biohazard_red",
    "memory_tampering",
    "planetary_embargo",
)

REVIEW_FLAGS = (
    "identity_conflict",
    "illegible_biometrics",
    "rescinded_denial",
    "sponsor_mismatch",
)

RISK_FLAGS = DISQUALIFYING_FLAGS + REVIEW_FLAGS

# Published in the field manual; the packets themselves name further revoked
# sponsors, which evidence.py picks up from the registry extract.
MANUAL_REVOKED_SPONSORS = frozenset({"SPN-0007", "SPN-0139", "SPN-4040"})

ADJUDICATIONS = ("APPROVED", "DENIED", "NEEDS_REVIEW")

# Raw classification points from EVALUATION.md, as payoff[predicted][truth].
# The -4 on a false approval is what makes DENIED the safe call whenever the
# APPROVED/DENIED margin is thin.
PAYOFF = {
    "APPROVED": {"APPROVED": 8.0, "DENIED": -4.0, "NEEDS_REVIEW": 1.0},
    "DENIED": {"APPROVED": 0.0, "DENIED": 8.0, "NEEDS_REVIEW": 1.0},
    "NEEDS_REVIEW": {"APPROVED": 2.0, "DENIED": 2.0, "NEEDS_REVIEW": 8.0},
}

# Schema-valid stand-ins; validate_submission.py rejects blanks and bad patterns.
UNKNOWN_TEXT = "unknown"
UNKNOWN_SPONSOR = "SPN-0000"
UNKNOWN_DATE = "1900-01-01"
