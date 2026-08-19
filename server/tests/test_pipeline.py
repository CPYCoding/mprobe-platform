"""
End-to-end regression coverage for the full run_pipeline() flow, using the
bundled server/samples/sample_dataset.zip as fixed test data.

The model runs in eval mode, so inference is deterministic — we verified
by hand that three separate runs against this exact dataset produced
byte-identical output zips. That determinism is what makes asserting on
exact counts here safe rather than flaky.

Needs the trained asset bundle (server/assets/) to exist — run
`python -m server.scripts.prepare_assets` first if these skip.
"""
import os
import statistics
import zipfile

import pytest

from server.pipeline import ASSET_DIR, run_pipeline

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_ZIP = os.path.join(ROOT_DIR, "server", "samples", "sample_dataset.zip")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ASSET_DIR, "manifest.json")) or not os.path.exists(SAMPLE_ZIP),
    reason="model assets or bundled sample dataset not present — run prepare_assets.py / make_sample_dataset.py first",
)


@pytest.fixture(scope="module")
def report_and_zip():
    with open(SAMPLE_ZIP, "rb") as f:
        zip_bytes = f.read()
    report, _job_id, output_path = run_pipeline(zip_bytes)
    with zipfile.ZipFile(output_path) as zf:
        kept_names = set(zf.namelist())
    return report, kept_names


def test_auto_removed_samples_are_excluded_from_the_cleaned_zip(report_and_zip):
    report, kept_names = report_and_zip
    materialized = report["detectors"][0]
    assert materialized["name"] == "materialized_trigger_detector"
    assert materialized["action"] == "auto_remove"

    auto_removed_ids = set(materialized["flagged_sample_ids"])
    assert auto_removed_ids  # sanity: the demo set does contain triggered samples
    assert auto_removed_ids.isdisjoint(kept_names)


def test_pending_review_samples_stay_in_the_cleaned_zip(report_and_zip):
    report, kept_names = report_and_zip
    blind = report["detectors"][1]
    assert blind["name"] == "blind_feature_detector"
    assert blind["action"] == "review"

    pending_ids = {entry["sample_id"] for entry in blind["flagged_sample_ids"]}
    assert pending_ids  # sanity: at this threshold the demo set does flag something
    assert pending_ids.issubset(kept_names)


def test_kept_count_matches_total_minus_auto_removed_only(report_and_zip):
    report, kept_names = report_and_zip
    assert len(kept_names) == report["kept_samples"]
    assert report["kept_samples"] == report["total_samples"] - report["auto_removed_samples"]


def test_materialized_flagged_ids_are_a_plain_string_list(report_and_zip):
    report, _kept_names = report_and_zip
    materialized_ids = report["detectors"][0]["flagged_sample_ids"]
    assert all(isinstance(x, str) for x in materialized_ids)


def test_blind_feature_flagged_entries_are_objects_sorted_by_score_desc(report_and_zip):
    report, _kept_names = report_and_zip
    entries = report["detectors"][1]["flagged_sample_ids"]

    assert all(isinstance(e, dict) for e in entries)
    assert all({"sample_id", "score", "level", "flagged_by"} <= e.keys() for e in entries)
    scores = [e["score"] for e in entries]
    assert scores == sorted(scores, reverse=True)


def test_blind_feature_level_is_a_median_split_of_this_run(report_and_zip):
    report, _kept_names = report_and_zip
    entries = report["detectors"][1]["flagged_sample_ids"]

    median_score = statistics.median(e["score"] for e in entries)
    for entry in entries:
        expected = "high" if entry["score"] >= median_score else "medium"
        assert entry["level"] == expected
