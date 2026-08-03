#!/usr/bin/env python3
"""Development harness: extract, score against train labels, show what missed."""

import argparse
import collections
import csv
import glob
import json
import multiprocessing as mp
import random
import sys
import time

sys.path.insert(0, "/w")

from mib import case, policy


def norm(field, value):
    value = " ".join(str(value or "").strip().split()).casefold()
    if field == "risk_flags":
        parts = sorted(p for p in value.split("|") if p and p != "none")
        return "|".join(parts) or "none"
    return value


def work(path):
    start = time.time()
    try:
        out = case.read_packet(path)
        result = (path, out["record"], out["features"])
    except Exception as exc:  # a broken packet must not stop the sweep
        result = (path, None, {"error": repr(exc)})
    elapsed = time.time() - start
    if elapsed > 25.0:
        print(f"SLOW {elapsed:6.1f}s {path}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    labels = {r["case_id"]: r for r in csv.DictReader(open("data/train_labels.csv"))}
    files = sorted(glob.glob("data/train/*.pdf"))
    if args.sample:
        random.Random(0).shuffle(files)
        files = files[: args.sample]
    else:
        files = files[args.offset : args.offset + args.limit]

    start = time.time()
    with mp.Pool(args.workers) as pool:
        results = []
        for item in pool.imap_unordered(work, files, chunksize=1):
            results.append(item)
            if len(results) % 100 == 0:
                print(f"... {len(results)}/{len(files)} in {time.time()-start:.0f}s", flush=True)
    elapsed = time.time() - start

    hit = collections.Counter()
    misses = collections.defaultdict(list)
    rows = []
    for path, record, features in results:
        cid = path.split("/")[-1][:-4]
        truth = labels[cid]
        if record is None:
            misses["error"].append((cid, features.get("error", "")))
            continue
        for f in case.OUTPUT_FIELDS:
            got, want = norm(f, record[f]), norm(f, truth[f])
            if got == want:
                hit[f] += 1
            else:
                misses[f].append((cid, got, want))
        rows.append({"case_id": cid, "record": record, "features": features, "truth": truth})

    n = len(files)
    print(f"{n} packets in {elapsed:.1f}s ({elapsed/n:.2f}s each, {args.workers} workers)")
    weights = {"applicant_name": 5, "species_code": 6, "home_world": 5, "visa_class": 5,
               "sponsor_id": 5, "arrival_date": 4, "declared_purpose": 3, "risk_flags": 8, "fee_status": 4}
    got_pts = sum(hit[f] * weights[f] for f in case.OUTPUT_FIELDS)
    max_pts = sum(n * weights[f] for f in case.OUTPUT_FIELDS)
    for f in case.OUTPUT_FIELDS:
        print(f"  {f:18s} {hit[f]:4d}/{n:4d}  {100*hit[f]/n:5.1f}%")
    print(f"  EXTRACTION ~ {50*got_pts/max_pts:.1f}/50")

    for f, items in misses.items():
        if not items:
            continue
        print(f"\n-- {f} misses ({len(items)}):")
        for item in items[: args.show]:
            print("   ", item)

    if rows:
        decisions = [policy.decide(r["record"], r["features"]) for r in rows]
        report(rows, decisions)
    if args.dump:
        with open(args.dump, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")


def report(rows, decisions):
    confusion = collections.Counter()
    raw = 0.0
    brier = []
    for row, (adj, conf) in zip(rows, decisions):
        truth = row["truth"]["adjudication"]
        confusion[(truth, adj)] += 1
        raw += {"APPROVED": {"APPROVED": 8, "DENIED": -4, "NEEDS_REVIEW": 1},
                "DENIED": {"APPROVED": 0, "DENIED": 8, "NEEDS_REVIEW": 1},
                "NEEDS_REVIEW": {"APPROVED": 2, "DENIED": 2, "NEEDS_REVIEW": 8}}[adj][truth]
        brier.append((conf - (1.0 if adj == truth else 0.0)) ** 2)
    n = len(rows)
    mean_brier = sum(brier) / len(brier)
    correct = sum(v for (t, p), v in confusion.items() if t == p)
    print(f"\n  accuracy {correct}/{n} = {100*correct/n:.1f}%")
    print(f"  CLASSIFICATION ~ {80*raw/(8*n):.1f}/80   CALIBRATION ~ {20*max(0,1-2*mean_brier):.1f}/20")
    print(f"  false approvals: {confusion[('DENIED','APPROVED')]}")
    for key in sorted(confusion):
        print(f"    {key[0]:12s} -> {key[1]:12s} {confusion[key]}")


if __name__ == "__main__":
    main()
