"""
Orchestrates the Dataset Cleaning & Poison Detection run.

Strictly sequential, not two independent scans merged afterward:
  1. materialized_trigger_detector runs on the FULL uploaded dataset.
  2. blind_feature_detector runs ONLY on the samples stage 1 did not flag.

The two stages carry different authority, and that difference is load
bearing, not cosmetic:
  - Stage 1 matches a KNOWN trigger signature. High confidence, so its
    hits are auto-removed from the cleaned dataset.
  - Stage 2 is a statistical anomaly signal (currently ~40% precision on
    held-out validation, per scripts/prepare_assets.py). It is NOT
    accurate enough to auto-delete on. Its hits are left in the cleaned
    dataset and reported as pending_review — a human decides, this
    pipeline doesn't. Silently merging stage 2 into "removed" auto-deletes
    on a coin-flip-grade signal, which is exactly the failure mode this
    split exists to prevent (verified against the bundled sample dataset:
    stage 2's only hit there was a false positive on a clean image).
"""
import json
import os
import uuid

import torch

from . import model as model_mod
from .dataset_io import load_dataset_from_zip, write_cleaned_zip
from .detectors.materialized_trigger import run_materialized_trigger_detector
from .detectors.blind_feature import run_blind_feature_detector

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_preprocess = model_mod.build_preprocess()

_state = {"model": None, "centroids": None, "threshold": None}


def _ensure_assets_loaded():
    if _state["model"] is not None:
        return
    manifest_path = os.path.join(ASSET_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(
            "Model assets not found. Run server/scripts/prepare_assets.py first "
            "to train the reference model and compute centroids/threshold."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    _state["model"] = model_mod.load_model(
        os.path.join(ASSET_DIR, "model_weights.pt"), _device
    )
    _state["centroids"] = torch.load(
        os.path.join(ASSET_DIR, "centroids.pt"), map_location=_device
    )
    _state["threshold"] = manifest["threshold"]


def run_pipeline(zip_bytes):
    _ensure_assets_loaded()

    samples = load_dataset_from_zip(zip_bytes)
    total = len(samples)

    materialized_report, materialized_flagged = run_materialized_trigger_detector(samples)

    remaining = [s for s in samples if s["id"] not in materialized_flagged]
    blind_report, blind_flagged = run_blind_feature_detector(
        remaining, _state["model"], _device, _state["centroids"], _state["threshold"], _preprocess
    )

    # Only stage 1 (known-signature match) removes samples from the
    # cleaned dataset. Stage 2 (statistical anomaly) flags for review but
    # is NOT accurate enough to delete on — its hits stay in the output.
    job_id = uuid.uuid4().hex
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_cleaned.zip")
    write_cleaned_zip(samples, materialized_flagged, output_path)

    report = {
        "total_samples": total,
        "auto_removed_samples": len(materialized_flagged),
        "pending_review_samples": len(blind_flagged),
        "kept_samples": total - len(materialized_flagged),
        "detectors": [materialized_report, blind_report],
    }
    return report, job_id, output_path
