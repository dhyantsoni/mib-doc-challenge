#!/usr/bin/env python3
"""Assert that nothing the hidden layer says ever reaches an emitted field.

The packets hide an answer key in white-on-white text whose extraction fields are
often nearly right but whose adjudication is always poisoned toward APPROVED.
This walks packets that carry one and checks two things: that the gate actually
caught the hidden span, and that no value the pipeline emits traces back to it
rather than to visible ink.
"""

import glob
import re
import sys

sys.path.insert(0, "/w")

from mib import case, page as pagelib

KEY = re.compile(r"MIB-\d{6}\s*,\s*(.+)")


def main(limit=120):
    checked = injected = leaked = 0
    poisoned_adjudications = 0

    for path in sorted(glob.glob("data/train/*.pdf"))[:limit]:
        pages = pagelib.load(path)
        hidden = [span.text for page in pages for span in page.hidden_spans]
        if not hidden:
            continue
        injected += 1

        payloads = [m.group(1) for text in hidden if (m := KEY.search(text))]
        if not payloads:
            continue
        checked += 1

        # Values the hidden key asserts, that visible evidence does not support.
        claimed = {part.strip().casefold() for payload in payloads for part in payload.split(",")}
        record = case.read_packet(path)["record"]
        visible = {
            value.casefold()
            for page in pages
            for span in page.visible_spans
            for value in [span.text]
        }
        visible_text = " ".join(visible)

        for name, value in record.items():
            if name in ("case_id", "risk_flags") or not value:
                continue
            text = str(value).casefold()
            if text in claimed and text not in visible_text:
                print(f"LEAK {path} {name}={value!r} appears only in hidden text")
                leaked += 1

        if "approved" in claimed and record.get("adjudication") == "APPROVED":
            poisoned_adjudications += 1

    print(f"packets with hidden spans: {injected}")
    print(f"packets carrying a hidden answer key: {checked}")
    print(f"emitted values traceable only to hidden text: {leaked}")
    print(f"adjudications matching a hidden APPROVED claim: {poisoned_adjudications}")
    return 1 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
