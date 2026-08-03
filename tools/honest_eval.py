#!/usr/bin/env python3
"""Score the pipeline the way the private test set will score it.

Evaluating a fitted model on the packets it was fitted on flatters it badly here:
in-sample the classifier reads 95% accurate against 76% held out, and the
catastrophic false-approval count reads 2 against 35. Both numbers matter more
than the average, so every score worth quoting comes from this file rather than
from a straight pass over the training set.

Extraction and the rule cascade have no sample dependence -- they score the same
either way. Only the residual model and its calibrator are cross-validated here.
"""

import argparse
import csv
import json
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/w")

from mib import policy, vocab

WEIGHTS = {"applicant_name": 5, "species_code": 6, "home_world": 5, "visa_class": 5,
           "sponsor_id": 5, "arrival_date": 4, "declared_purpose": 3, "risk_flags": 8,
           "fee_status": 4}


def norm(field, value):
    value = " ".join(str(value or "").strip().split()).casefold()
    if field == "risk_flags":
        return "|".join(sorted(p for p in value.split("|") if p and p != "none")) or "none"
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/train_cache.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    labels = {r["case_id"]: r for r in csv.DictReader(open("data/train_labels.csv"))}
    rows = [r for r in (json.loads(l) for l in open(args.cache)) if r["case_id"] in labels]
    truth = np.array([labels[r["case_id"]]["adjudication"] for r in rows])

    X = np.array([policy.vectorize(r["record"], r["features"]) for r in rows])
    modelled = np.array([r["features"].get("finding", "") not in vocab.ADJUDICATIONS for r in rows])

    from tools.train import build_classifier

    held = np.zeros((len(rows), 3))
    classes = None
    for train_idx, test_idx in StratifiedKFold(args.folds, shuffle=True, random_state=0).split(
        X[modelled], truth[modelled]
    ):
        fitted = build_classifier().fit(X[modelled][train_idx], truth[modelled][train_idx])
        classes = list(fitted.classes_)
        held[np.flatnonzero(modelled)[test_idx]] = fitted.predict_proba(X[modelled][test_idx])

    actions, raws, outcomes = [], [], []
    for index, row in enumerate(rows):
        if modelled[index]:
            probability = dict(zip(classes, held[index]))
        else:
            finding = row["features"]["finding"]
            probability = {n: (0.97 if n == finding else 0.015) for n in vocab.ADJUDICATIONS}
        action, raw = policy.act(row["record"], row["features"], probability)
        actions.append(action)
        raws.append(raw)
        outcomes.append(float(action == truth[index]))

    raws, outcomes = np.array(raws), np.array(outcomes)
    calibrated = np.zeros(len(rows))
    for train_idx, test_idx in StratifiedKFold(args.folds, shuffle=True, random_state=1).split(
        raws.reshape(-1, 1), outcomes
    ):
        fitted = IsotonicRegression(y_min=0.02, y_max=0.99, out_of_bounds="clip")
        calibrated[test_idx] = fitted.fit(raws[train_idx], outcomes[train_idx]).predict(raws[test_idx])

    recovered = sum(
        WEIGHTS[f] for r in rows for f in WEIGHTS
        if norm(f, r["record"][f]) == norm(f, labels[r["case_id"]][f])
    )
    extraction = 50 * recovered / (len(rows) * sum(WEIGHTS.values()))
    classification = 80 * sum(
        vocab.PAYOFF[a][t] for a, t in zip(actions, truth)
    ) / (8 * len(rows))
    brier = float(np.mean((calibrated - outcomes) ** 2))
    calibration = 20 * max(0.0, 1 - 2 * brier)
    false_approvals = sum(1 for a, t in zip(actions, truth) if a == "APPROVED" and t == "DENIED")

    print(f"cases {len(rows)}  ({int(modelled.sum())} decided with the model)")
    for field in WEIGHTS:
        hit = sum(1 for r in rows if norm(field, r["record"][field]) == norm(field, labels[r["case_id"]][field]))
        print(f"  {field:18s} {hit:4d}/{len(rows)}  {100*hit/len(rows):5.1f}%")
    print(f"EXTRACTION {extraction:.1f}/50  CLASSIFICATION {classification:.1f}/80  "
          f"CALIBRATION {calibration:.1f}/20  TOTAL {extraction+classification+calibration:.1f}/150")
    print(f"accuracy {outcomes.mean():.4f}   catastrophic false approvals {false_approvals}")


if __name__ == "__main__":
    main()
