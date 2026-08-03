#!/usr/bin/env python3
"""Fit the residual adjudication model and its confidence calibrator.

Features come from the pipeline's own extractions, not from the gold fields, so
the model sees the same noisy input at training time that it will see at scoring
time — including the packets where OCR failed.

Writes mib/model.joblib, which the image loads at runtime. The artifact is
produced here and shipped inside the image; it is never loaded from input data.
"""

import argparse
import csv
import glob
import json
import multiprocessing as mp
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/w")

from mib import case, policy, vocab


def work(path):
    try:
        out = case.read_packet(path)
        return {"case_id": out["record"]["case_id"], "record": out["record"], "features": out["features"]}
    except Exception:
        return None


def build_cache(workers, cache_path):
    files = sorted(glob.glob("data/train/*.pdf"))
    with mp.Pool(workers) as pool:
        rows = [row for row in pool.imap_unordered(work, files, chunksize=2) if row]
    with open(cache_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/train_cache.jsonl")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--out", default="mib/model.joblib")
    args = ap.parse_args()

    if args.rebuild:
        rows = build_cache(args.workers, args.cache)
    else:
        rows = [json.loads(line) for line in open(args.cache)]

    labels = {r["case_id"]: r for r in csv.DictReader(open("data/train_labels.csv"))}
    rows = [row for row in rows if row["case_id"] in labels]

    X = np.array([policy.vectorize(row["record"], row["features"]) for row in rows])
    y = np.array([labels[row["case_id"]]["adjudication"] for row in rows])
    # A note that states the finding is handled by a rule, and its packets would
    # otherwise teach the model to lean on features it will not have elsewhere.
    mask = np.array([row["features"].get("finding", "") not in vocab.ADJUDICATIONS for row in rows])

    def fresh():
        return HistGradientBoostingClassifier(
            max_depth=4, max_iter=260, learning_rate=0.07, l2_regularization=1.0,
            min_samples_leaf=12, random_state=0,
        )

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    held_probability = np.zeros((len(rows), 3))
    for train_index, test_index in folds.split(X[mask], y[mask]):
        model = fresh().fit(X[mask][train_index], y[mask][train_index])
        held_probability[np.flatnonzero(mask)[test_index]] = model.predict_proba(X[mask][test_index])

    classifier = fresh().fit(X[mask], y[mask])
    order = list(classifier.classes_)

    # Calibrate P(the action we would emit is correct) against what actually
    # happened on held-out folds -- exactly the quantity the Brier section scores.
    confidences, outcomes = [], []
    for index, row in enumerate(rows):
        probability = dict(zip(order, held_probability[index]))
        if not mask[index]:
            finding = row["features"]["finding"]
            probability = {name: (0.97 if name == finding else 0.015) for name in vocab.ADJUDICATIONS}
        action, raw = _act(row["record"], row["features"], probability)
        confidences.append(raw)
        outcomes.append(float(action == y[index]))

    calibrator = IsotonicRegression(y_min=0.02, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(confidences, outcomes)

    joblib.dump({"classifier": classifier, "calibrator": calibrator}, args.out)

    raw_brier = float(np.mean([(c - o) ** 2 for c, o in zip(confidences, outcomes)]))
    fitted = calibrator.predict(confidences)
    cal_brier = float(np.mean([(c - o) ** 2 for c, o in zip(fitted, outcomes)]))
    accuracy = float(np.mean(outcomes))
    print(f"cases={len(rows)} model_cases={int(mask.sum())} held-out accuracy={accuracy:.4f}")
    print(f"brier raw={raw_brier:.4f} calibrated={cal_brier:.4f} -> calibration {20*max(0,1-2*cal_brier):.1f}/20")


def _act(record, features, probability):
    forced = policy.constrain(record, features)
    if forced:
        probability = {n: 0.90 * (n == forced) + 0.10 * probability.get(n, 0.0) for n in vocab.ADJUDICATIONS}
    total = sum(probability.values()) or 1.0
    probability = {n: v / total for n, v in probability.items()}
    action = max(
        vocab.ADJUDICATIONS,
        key=lambda choice: sum(vocab.PAYOFF[choice][t] * probability[t] for t in vocab.ADJUDICATIONS),
    )
    return action, probability[action]


if __name__ == "__main__":
    main()
