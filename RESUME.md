# Where this is, and how to pick it up

Everything needed to continue is either committed to git or regenerable by the
commands below. Nothing important lives in `/tmp`.

## State

| | |
| --- | --- |
| Solution code | `main` branch, repo root (`Dockerfile`, `mib/`, `tools/`) |
| Trained model | `mib/model.joblib`, committed (~5 KB) |
| Submission folder | `submission` branch: `submissions/dhyantsoni/` only |
| Honest score | **122.5/150** — extraction 41.1, classification 65.6, calibration 15.8 |
| Held-out accuracy | 81.4%, 35 catastrophic false approvals |
| Runtime | ~1.6 s per PDF at 6 CPUs, against a 6 s budget |

Scores come from `tools/honest_eval.py`, which cross-validates the fitted parts.
A plain pass over the training set reads about 134 and is not a real number.

## Rebuilding from nothing

```bash
# 1. data (2.9 GB, checksum in data/README.md)
curl -sL -o mib-data.zip \
  https://huggingface.co/datasets/arjun-krishna1/mib-doc-challenge-data/resolve/main/mib-doc-challenge-public-data-v2026-07-07.zip
unzip -q mib-data.zip -d .

# 2. image
docker build -t mib-submission .

# 3. features + model (~35 min on 6 cores; writes mib/model.joblib)
docker run --rm --cpus 6 -v "$PWD":/w -v "$PWD/work":/out -w /w \
  -e PYTHONPATH=/w -e OMP_THREAD_LIMIT=1 --entrypoint python3 mib-submission \
  tools/train.py --rebuild --workers 6 --cache /out/train_cache.jsonl

# 4. score honestly
docker run --rm --cpus 4 -v "$PWD":/w -v "$PWD/work":/out -w /w \
  -e PYTHONPATH=/w --entrypoint python3 mib-submission \
  tools/honest_eval.py --cache /out/train_cache.jsonl
```

`work/` is gitignored and holds the feature cache; step 3 regenerates it.

## The validation predictions

The run is **resumable**: `mib/main.py` reads any predictions already in the
output file, discards a half-written trailing line, and processes only the cases
that are missing. Re-issuing the same command after an interruption costs only
the remainder, and the same mechanism is what makes the scoring runtime's own
timeout survivable.

```bash
mkdir -p work/val
docker run --rm --network none --cpus 6 --memory 8g --pids-limit 512 --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --mount type=bind,src="$PWD/data/validation",dst=/input,readonly \
  --mount type=bind,src="$PWD/work/val",dst=/output \
  mib-submission /input /output/predictions.jsonl

python3 scripts/validate_submission.py \
  --submission work/val/predictions.jsonl --manifest data/validation_manifest.csv
```

Expect ~2.5 hours for 5,000 packets on 6 cores. Check progress with
`wc -l work/val/predictions.jsonl`.

## Publishing

Nothing has been pushed. When ready:

```bash
git push origin main
git push origin submission
gh pr create --repo 8090-inc/mib-doc-challenge --head dhyantsoni:submission --base main
```

`main` must go too — `submissions/dhyantsoni/SUBMISSION.md` links to it. The
Google form in the challenge README is required alongside the pull request.

To rebuild the submission branch after regenerating predictions:

```bash
git switch --detach 38ce888 && git switch -c submission-new
mkdir -p submissions/dhyantsoni
git show main:submissions/dhyantsoni/MEMO.md > submissions/dhyantsoni/MEMO.md
git show main:submissions/dhyantsoni/SUBMISSION.md > submissions/dhyantsoni/SUBMISSION.md
cp work/val/predictions.jsonl submissions/dhyantsoni/predictions.jsonl
git add submissions/dhyantsoni && git commit -m "Add submission for dhyantsoni"
```

## What was already tried and rejected

Recorded so it is not re-attempted: 300 DPI rendering (worse than 220 — the
embedded scans are 144 DPI), morphological removal of scan streaks (deletes
letter stems; page classification 110 → 96), ink-bbox cropping, 1.5x upscaling,
re-reading value strips at 3x under `psm 7` (+0.5%), feature-aware calibration
(no gain over 1-D isotonic), and class weighting or regularisation sweeps to cut
false approvals (expected-utility selection is already optimal for the
probabilities it is given).
