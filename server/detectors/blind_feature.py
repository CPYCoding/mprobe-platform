"""
Stage 2 — blind feature detector.

Runs only on the samples stage 1 (materialized_trigger_detector) did NOT
flag. Uses the bundled, pre-trained feature extractor plus its precomputed
per-class centroids + anomaly threshold (all computed offline at listing
time from the seller's own trusted reference data — see
scripts/prepare_assets.py) to score how far each remaining sample sits from
where a "normal" sample of its predicted class should land in feature
space. Catches poisoning that doesn't match any known trigger signature.

This signal is advisory, not a deletion trigger: it's a statistical score,
not a known-signature match, and validated accuracy on held-out data is
~40% — nowhere near good enough to auto-remove on. The pipeline (see
pipeline.py) keeps flagged_sample_ids in the cleaned dataset and reports
them under pending_review; only a human calls the final judgment on them.

flagged_sample_ids here is a list of objects (sample_id/score/level/
flagged_by), NOT the plain string list materialized_trigger_detector
returns — the two detectors carry different authority (auto_remove vs
review), so their shapes are intentionally different. Callers must branch
on a detector's `action` field rather than assume a shape.

`level` (high/medium) is computed here, once, so any consumer of this
report — this page or a future direct API caller — sees the same label.
It's a median split of THIS run's flagged scores, not a fixed multiple of
the threshold: raw distance values are only meaningful relative to each
other for a given feature extractor, and swapping in a different backbone
later would silently invalidate a hardcoded cutoff. Ties at the median
land in "high" — that only skews the display label, never what gets
removed, so it's left unhandled on purpose.
"""
import statistics

import torch

BATCH_SIZE = 64


def _score_batch(model, device, centroids, batch_tensor):
    with torch.no_grad():
        features = model.features(batch_tensor)
        features = torch.flatten(features, 1)
        logits = model.classifier(features)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_labels = probs.max(dim=1)
        distances = torch.cdist(features, centroids)
        pred_dist = distances[torch.arange(distances.size(0), device=device), pred_labels]
        # pred_dist_plus_uncertainty: far from its own class centroid AND
        # low confidence is the strongest anomaly signal.
        return pred_dist + (1.0 - confidence)


def run_blind_feature_detector(samples, model, device, centroids, threshold, preprocess):
    """
    samples: list of {"id": str, "image": PIL.Image (RGB)} — already
    filtered down to what stage 1 left untouched.
    Returns (report_dict, flagged_id_set).
    """
    flagged = []  # [(sample_id, score), ...]

    for start in range(0, len(samples), BATCH_SIZE):
        chunk = samples[start:start + BATCH_SIZE]
        batch_tensor = torch.stack([preprocess(s["image"]) for s in chunk]).to(device)
        scores = _score_batch(model, device, centroids, batch_tensor)
        for sample, score_value in zip(chunk, scores.detach().cpu().tolist()):
            if score_value >= threshold:
                flagged.append((sample["id"], score_value))

    flagged_entries = []
    if flagged:
        median_score = statistics.median(score for _, score in flagged)
        for sample_id, score_value in flagged:
            flagged_entries.append({
                "sample_id": sample_id,
                "score": score_value,
                "level": "high" if score_value >= median_score else "medium",
                "flagged_by": "blind_feature_detector",
            })
        flagged_entries.sort(key=lambda e: e["score"], reverse=True)

    report = {
        "name": "blind_feature_detector",
        "confidence": "advisory_only",
        "action": "review",
        "description": (
            "Statistical anomaly vs. clean feature baseline — recommended for "
            "human review, not automatic removal"
        ),
        "flagged_count": len(flagged_entries),
        "flagged_sample_ids": flagged_entries,
    }
    return report, {e["sample_id"] for e in flagged_entries}
