# MProbe Platform

Front-end + backend for **MProbe** — a security-certified marketplace for ML models and datasets. Anyone can list a model or dataset; every listing is scanned by MProbe (integrity + backdoor/poison detection) before it goes live.

## Status at a glance

Most of this site is a **static front-end prototype** — prices, ratings, and inference results on `models/person-detector.html` / `models/red-car-detector.html` are placeholders generated in the browser with `Math.random()` (see `models/run.js`). `upload.html`'s "list a model" flow is a simulated scan; it doesn't call anything real yet.

**One exception: `models/dataset-cleaning.html` is wired to a real, working backend** (`server/`). It's the only listing on the site that actually runs a trained model and does real work — a labeled image dataset in, a two-stage poisoned-sample scan, a cleaned dataset + report out. Everything below this point is about that backend.

## Quick start

```bash
# 1. install deps (from repo root)
pip install -r server/requirements.txt

# 2. prepare offline model assets (trains the reference model, computes
#    per-class centroids + anomaly threshold — one-time, ~a few minutes,
#    downloads CIFAR-10 the first time)
python -m server.scripts.prepare_assets

# 3. build the bundled demo dataset (mix of clean / known-trigger /
#    novel-anomaly images, used by the "load a sample dataset" button)
python -m server.scripts.make_sample_dataset

# 4. run the backend
uvicorn server.main:app --reload --port 8001

# 5. open models/dataset-cleaning.html in a browser
#    (the port in its window.MODEL.backend must match step 4)

# 6. run the regression tests
python -m pytest server/tests
```

Steps 2–3 only need to be re-run when the model/asset-generation code changes — the trained assets (`server/assets/`) and demo zip (`server/samples/`) are checked into the repo, not regenerated on every run.

## Repo structure

```
mprobe-platform/
├── marketplace.html          # main store + profile (front-end mock)
├── upload.html                # "list a model" flow (front-end mock, simulated scan)
├── models/
│   ├── person-detector.html   # mock detector Run page
│   ├── red-car-detector.html  # mock detector Run page
│   ├── dataset-cleaning.html  # REAL Run page — talks to server/ via fetch()
│   ├── run.css                 # shared look for all Run pages
│   └── run.js                  # shared behaviour; renderLiveCleaning()/
│                                # liveCleanDone() is the real-backend path,
│                                # everything else in this file is simulated
├── server/                    # the real dataset-cleaning backend
│   ├── main.py                 # FastAPI app — 3 endpoints (below)
│   ├── pipeline.py             # orchestrates the two-stage scan
│   ├── model.py                 # feature-extractor model + preprocessing
│   ├── dataset_io.py            # zip in / cleaned zip out
│   ├── detectors/
│   │   ├── materialized_trigger.py  # stage 1: known-signature match
│   │   └── blind_feature.py         # stage 2: statistical anomaly
│   ├── scripts/
│   │   ├── prepare_assets.py        # offline: train + calibrate (run once)
│   │   ├── calibrate_threshold.py   # recompute just the threshold, no retrain
│   │   └── make_sample_dataset.py   # builds the bundled demo dataset
│   ├── tests/                   # pytest regression suite (see below)
│   ├── assets/                  # trained model + centroids + manifest (checked in)
│   ├── samples/                 # bundled demo dataset (checked in)
│   └── .data/, outputs/         # gitignored — CIFAR-10 cache, per-run job zips
└── docs/
    └── data-cleaning-v2-design.md   # full design rationale — read this
        before changing anything architectural in server/
```

## Why `dataset-cleaning.html` needs a backend running

Every other Run page on the site fakes its results with plain JavaScript — no real computation happens, so nothing needs to be running besides the browser. Dataset cleaning is different: it runs an actual trained model (feature extraction, distance-to-centroid scoring), which a browser can't do on its own. So the architecture is client/server:

```
models/dataset-cleaning.html  --fetch()-->  server/main.py (FastAPI, port 8001)
   (static page, no server needed)              (must be running for the page to work)
```

In production this backend would run on a real always-on server (e.g. `api.mprobe.ai`) and visitors wouldn't need to do anything. Locally, `uvicorn ... --port 8001` on your own machine is standing in for that.

## How the dataset-cleaning service works

Full rationale is in `docs/data-cleaning-v2-design.md` — this is the short version.

**Two-stage pipeline, strictly sequential, not two independent scans merged:**

1. **`materialized_trigger_detector`** (stage 1) — direct pixel-level match against a registry of *known* trigger signatures (currently one: a solid red patch injected in the bottom-right corner, scaled to image size). No training involved, like an antivirus signature list. Runs on the **full** uploaded dataset. High confidence → its hits are **auto-removed** from the cleaned dataset.
2. **`blind_feature_detector`** (stage 2) — runs only on what stage 1 did *not* flag. Uses a pre-trained, frozen feature extractor + precomputed per-class centroids to score how far each sample sits from where a "normal" sample of its class should land in feature space. Catches poisoning that doesn't match any known signature — but it's a statistical signal, currently ~40% precision on held-out validation. **Not accurate enough to auto-delete on.** Its hits stay in the downloadable cleaned zip and are reported as `pending_review`, sorted by score with a `high`/`medium` severity label (a median split of that run's own scores, not a fixed cutoff — stays meaningful even if the underlying model changes) — a human makes the final call, the pipeline doesn't.

This split is load-bearing, not cosmetic — an earlier version merged both stages' hits into one auto-delete set, which silently deleted good data on a coin-flip-grade signal. `server/tests/test_pipeline.py` locks this in so it can't regress.

**Why training happens offline, once, not per-request:** an earlier version retrained a model from scratch on every "run cleaning" click. That measured out at 10+ minutes per run and swung from 0% to 63.6% accuracy on nothing but a different random seed — unusable. `server/scripts/prepare_assets.py` now trains once, offline, and the pipeline only ever loads and does inference on that fixed asset bundle.

**Report shape** (`report_and_zip` fixture in the tests shows this best): `total_samples`, `auto_removed_samples`, `pending_review_samples`, `kept_samples`, and a `detectors` list. Each detector reports `action` (`auto_remove` or `review`) and `flagged_sample_ids` — but the *shape* of that list intentionally differs by detector: `materialized_trigger_detector`'s is a plain list of filename strings; `blind_feature_detector`'s is a list of `{sample_id, score, level, flagged_by}` objects. Frontend code must branch on `action`, never assume a shape.

## Testing

```bash
python -m pytest server/tests
```

12 tests, ~5s, using the trained assets + bundled sample dataset as fixed test data (the model runs in eval mode, so inference is deterministic — verified by hand that repeated runs produce byte-identical output). Assertions are structural (`auto_removed` samples are excluded from the cleaned zip, `pending_review` samples aren't, scores are sorted descending, `level` matches the median split) rather than hardcoded to specific counts, so they should keep passing across model/threshold changes without needing updates. Tests involving the trained model auto-skip if `server/assets/` or `server/samples/sample_dataset.zip` don't exist yet.

## Known limitations (V1)

- **Feature extractor only understands CIFAR-10-like images.** `model.py`'s `SmallCifarClassifier` was trained from scratch on CIFAR-10's 10 classes at 32×32. A buyer uploading a dataset outside that domain (medical images, faces, arbitrary categories) gets a `blind_feature` layer that's effectively noise. **In progress:** swapping to a frozen, pretrained general backbone (`torchvision.models.resnet18`, ImageNet weights) *and* switching class grouping from the model's own predicted label to the buyer's own folder-declared label — both changes are needed together to actually fix this; the backbone swap alone won't be demonstrable as an improvement since all current test/demo data is CIFAR-10.
- **Audit report is a simplified version of the target design.** `docs/data-cleaning-v2-design.md` §5 specifies `scan_id`, `dataset_hash`, per-sample `recommended_action`, and an Ed25519 `report_signature` for tamper-evidence — none of that is built yet. Deliberately deprioritized versus the two points above.
- **`upload.html`'s "list a model" flow is still a front-end simulation.** Real sellers can't actually upload a dataset and get `server/assets/`-style assets generated for their own listing yet — today, asset prep is a manual script run by whoever maintains this repo.
- **`server/outputs/` has no cleanup.** Every `/run` call writes a new `{job_id}_cleaned.zip` there and nothing ever deletes it.
- **No CI.** Tests exist and pass locally but nothing runs them automatically on push.

## Editing a mock model's Run page

Each `models/*.html` (other than `dataset-cleaning.html`) has a `window.MODEL` config block at the top — change the title, dataset, sample frames, and text there. Shared styling lives in `run.css`, shared behaviour in `run.js`. `dataset-cleaning.html`'s config additionally has `live: true` and `backend: "http://127.0.0.1:8001"` — that's what routes it to the real server instead of the simulated path in `run.js`.
