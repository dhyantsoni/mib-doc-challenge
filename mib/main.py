"""Entry point: read a directory of packets, write one JSONL prediction per PDF.

Results are appended as they finish, so a container stopped at the runtime limit
is still scored on everything it had already decided.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from . import case, policy, vocab

BUDGET_SECONDS_PER_PDF = 6.0
_deadline = 0.0


def _predict(path: str) -> dict:
    fast = _deadline and time.time() > _deadline
    try:
        packet = case.read_packet(path, allow_ocr=not fast)
        record = packet["record"]
        adjudication, confidence = policy.decide(record, packet["features"])
    except Exception:
        # Never drop a case: a bare NEEDS_REVIEW at low confidence still scores
        # better than the missing-case penalty plus a forfeited extraction row.
        record = {
            "case_id": Path(path).stem,
            "applicant_name": vocab.UNKNOWN_TEXT,
            "species_code": vocab.UNKNOWN_TEXT,
            "home_world": vocab.UNKNOWN_TEXT,
            "visa_class": vocab.UNKNOWN_TEXT,
            "sponsor_id": vocab.UNKNOWN_SPONSOR,
            "arrival_date": vocab.UNKNOWN_DATE,
            "declared_purpose": vocab.UNKNOWN_TEXT,
            "risk_flags": "none",
            "fee_status": "unknown",
        }
        adjudication, confidence = "NEEDS_REVIEW", 0.2

    return {**record, "adjudication": adjudication, "confidence": confidence}


def _cpu_quota() -> int:
    """Workers the container may actually run, not cores the host happens to have.

    os.cpu_count() reports the host's cores even under `--cpus 4`, and
    oversubscribing tesseract past the real quota costs an order of magnitude in
    throughput rather than a few percent.
    """
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _init(deadline: float) -> None:
    global _deadline
    _deadline = deadline


def main(input_dir: str, output_path: str) -> None:
    pdfs = sorted(str(p) for p in Path(input_dir).glob("*.pdf"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    workers = _cpu_quota()
    deadline = time.time() + 0.85 * BUDGET_SECONDS_PER_PDF * max(1, len(pdfs))

    with open(output, "w") as handle:
        if not pdfs:
            return
        with mp.Pool(workers, initializer=_init, initargs=(deadline,)) as pool:
            for prediction in pool.imap_unordered(_predict, pdfs, chunksize=1):
                handle.write(json.dumps(prediction, sort_keys=True) + "\n")
                handle.flush()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m mib.main <input_pdf_dir> <output_path>")
    main(sys.argv[1], sys.argv[2])
