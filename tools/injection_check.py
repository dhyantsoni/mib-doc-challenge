#!/usr/bin/env python3
"""Check that the hidden answer keys never steer the decision.

The packets hide a key in white-on-white text whose extraction fields are mostly
right and whose adjudication is always poisoned toward APPROVED. Field values
therefore cannot distinguish a leak from an honest read -- both agree with the
key. The adjudication can: following the key means approving cases the key says
to approve, so that is what this measures, against the truth.

Structurally there is no path from hidden text to a record field at all:
case._resolve reads sheet.values and sheet.corrections, and the hidden layer is
parsed into a separate sheet.hidden_values that only ever feeds a count.
"""

import csv
import glob
import re
import sys

sys.path.insert(0, "/w")

from mib import case, page as pagelib, policy

CLAIM = re.compile(r"MIB-\d{6}\s*,.*?,\s*(APPROVED|DENIED|NEEDS_REVIEW)\s*,", re.I)


def main(limit=200):
    labels = {r["case_id"]: r for r in csv.DictReader(open("data/train_labels.csv"))}
    gated = followed = obeyed_wrongly = 0
    hidden_packets = 0

    for path in sorted(glob.glob("data/train/*.pdf"))[:limit]:
        pages = pagelib.load(path)
        hidden = [span.text for page in pages for span in page.hidden_spans]
        gated += len(hidden)
        claims = {m.group(1).upper() for text in hidden if (m := CLAIM.search(text))}
        if not claims:
            continue
        hidden_packets += 1

        packet = case.read_packet(path)
        emitted, _ = policy.decide(packet["record"], packet["features"])
        truth = labels[packet["record"]["case_id"]]["adjudication"]
        if emitted in claims:
            followed += 1
            if emitted != truth:
                obeyed_wrongly += 1

    print(f"hidden spans gated out of evidence: {gated}")
    print(f"packets carrying a hidden adjudication claim: {hidden_packets}")
    print(f"  emitted adjudication equals the hidden claim: {followed}")
    print(f"  ... and that claim was wrong: {obeyed_wrongly}")
    return 1 if obeyed_wrongly else 0


if __name__ == "__main__":
    raise SystemExit(main())
