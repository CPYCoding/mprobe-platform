"""
Regression coverage for stage 1 (materialized_trigger_detector).

Kept deliberately small: this locks in behavior we've already verified by
hand more than once, so the upcoming feature-extractor swap (which doesn't
touch this file at all) can't silently regress it without a test failing.
"""
import numpy as np
import pytest
from PIL import Image

from server.detectors.materialized_trigger import (
    TRIGGER_REGISTRY,
    _patch_size,
    run_materialized_trigger_detector,
)

RED_SQUARE = TRIGGER_REGISTRY[0]


def _make_image(width, height, with_trigger):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    if with_trigger:
        patch_w, patch_h = _patch_size(width, height, RED_SQUARE)
        arr[height - patch_h:height, width - patch_w:width] = RED_SQUARE["color"]
    return Image.fromarray(arr)


@pytest.mark.parametrize("width,height", [(32, 32), (200, 150), (64, 64), (500, 333)])
def test_flags_only_the_triggered_image_across_sizes(width, height):
    clean_id = f"clean_{width}x{height}.png"
    triggered_id = f"triggered_{width}x{height}.png"
    samples = [
        {"id": clean_id, "image": _make_image(width, height, with_trigger=False)},
        {"id": triggered_id, "image": _make_image(width, height, with_trigger=True)},
    ]

    _report, flagged = run_materialized_trigger_detector(samples)

    assert flagged == {triggered_id}


def test_report_shape_stays_a_plain_string_list():
    # Guards against ever accidentally reshaping this into an object list
    # the way blind_feature_detector's flagged_sample_ids is — the
    # frontend branches on d.action assuming these two shapes never merge.
    samples = [{"id": "a.png", "image": _make_image(32, 32, with_trigger=True)}]

    report, _flagged = run_materialized_trigger_detector(samples)

    assert report["action"] == "auto_remove"
    assert report["flagged_sample_ids"] == ["a.png"]
    assert all(isinstance(x, str) for x in report["flagged_sample_ids"])


def test_no_false_positive_on_a_clean_image():
    samples = [{"id": "clean.png", "image": _make_image(32, 32, with_trigger=False)}]

    _report, flagged = run_materialized_trigger_detector(samples)

    assert flagged == set()
