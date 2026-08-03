# Solution — offline intake pipeline

This fork adds a Dockerised pipeline that reads a directory of PDF case packets and
writes `predictions.jsonl`. Everything below the challenge's own files is the
solution; nothing in the original challenge material was modified.

```bash
docker build -t mib-submission .
docker run --rm --network none \
  --mount type=bind,src="$PWD/data/train",dst=/input,readonly \
  --mount type=bind,src="/tmp/mib-output",dst=/output \
  mib-submission /input /output/predictions.jsonl
```

## Layout

| Path | Role |
| --- | --- |
| `mib/page.py` | renders each page and verifies that every text span leaves ink, separating visible evidence from hidden text |
| `mib/ocr.py` | Tesseract recovery for the ~47% of pages that arrive as degraded scans |
| `mib/extract.py` | template- and label-anchored reading, closed-vocabulary snapping |
| `mib/case.py` | evidence ledger, precedence resolution, case-id scoping |
| `mib/policy.py` | manual rules as constraints, learned residual, expected-utility decision |
| `mib/main.py` | CLI and process pool |
| `tools/train.py` | rebuilds `mib/model.joblib` from `data/train/` |
| `tools/score.py` | local scoring harness against `data/train_labels.csv` |

`submissions/dhyantsoni/MEMO.md` explains why each piece works the way it does.

## Rebuilding the model

`mib/model.joblib` is committed so the image builds without the dataset present.
To regenerate it, with the public data unpacked at the repository root:

```bash
python3 tools/train.py --rebuild --workers 4
```

## Runtime notes

No network, no GPU, no API keys, and no LLM or VLM at any point. The only model
artifacts are Tesseract's bundled English data and a gradient-boosted classifier
under 1 MiB trained here on the public training split. Worker count is read from
the container's CPU quota rather than the host core count, because oversubscribing
Tesseract past the real quota costs an order of magnitude in throughput.
