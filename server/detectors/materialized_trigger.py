"""
Stage 1 — materialized trigger detector.

Direct pixel-level match against a registry of KNOWN trigger signatures.
This runs on the full uploaded dataset first: it's cheap, high-confidence,
and only catches triggers we already have a definition for (like an
anti-virus signature list) — anything novel falls through to stage 2
(blind_feature_detector), which only ever sees what this stage did NOT flag.

To add a new known trigger later, append a definition to TRIGGER_REGISTRY
and implement its "type" in _MATCHERS if it's a new kind of pattern.
"""
import numpy as np

TRIGGER_REGISTRY = [
    {
        "id": "red_square_v1",
        "description": "Solid red patch injected at the bottom-right corner",
        "type": "solid_color_corner_patch",
        "color": (255, 0, 0),
        "corner": "bottom_right",
        # Spec: at a 128x128 reference size the patch is 10x10px, scaled
        # proportionally to the image's actual dimensions, 5px minimum.
        "ref_dim": 128,
        "ref_patch": 10,
        "min_patch": 5,
        "color_tolerance": 12,   # per-channel max abs diff to count as a matching pixel
        "match_ratio": 0.9,      # fraction of the patch that must match to flag the image
    },
    {
        "id": "backdoor_grid_v1",
        "description": "Fixed black/white grid pattern from the original federated-learning backdoor-attack research",
        "type": "fixed_grid_pattern",
        # Ported from data_washing_experiment/clean_reference_detector.py's
        # BACKDOOR_PATTERN / BACKDOOR_X_TOP / BACKDOOR_Y_TOP / MASK_VALUE —
        # this is the actual pattern used to poison training images in that
        # codebase's attack simulation, not a pattern we invented. None =
        # masked position (MASK_VALUE there): leave the underlying pixel
        # alone, don't check or draw it. Reference frame is a 32x32 image;
        # position and cell size scale proportionally to the image's actual
        # dimensions, same idea as red_square_v1's scaling.
        "grid": [
            [255, 0, 255],
            [None, 255, None],
            [None, None, 0],
            [None, 255, None],
            [255, 0, 255],
        ],
        "ref_dim": 32,
        "ref_row_top": 3,
        "ref_col_top": 23,
        "color_tolerance": 12,
        "match_ratio": 0.85,   # 9 checkable cells in this pattern — allow one to miss
    },
]


def _patch_size(width, height, trigger):
    patch_w = max(int(trigger["ref_patch"] * width / trigger["ref_dim"]), trigger["min_patch"])
    patch_h = max(int(trigger["ref_patch"] * height / trigger["ref_dim"]), trigger["min_patch"])
    return min(patch_w, width), min(patch_h, height)


def _corner_slice(width, height, patch_w, patch_h, corner):
    if corner == "bottom_right":
        return slice(height - patch_h, height), slice(width - patch_w, width)
    raise ValueError(f"Unsupported corner: {corner}")


def _match_solid_color_corner_patch(image_rgb, trigger):
    width, height = image_rgb.size
    patch_w, patch_h = _patch_size(width, height, trigger)
    row_slice, col_slice = _corner_slice(width, height, patch_w, patch_h, trigger["corner"])

    arr = np.asarray(image_rgb)
    patch = arr[row_slice, col_slice, :3]
    color = np.array(trigger["color"], dtype=np.int16)
    diff = np.abs(patch.astype(np.int16) - color)
    matches = np.all(diff <= trigger["color_tolerance"], axis=-1)
    return float(matches.mean()) if matches.size else 0.0


def _grid_cell_bounds(width, height, trigger, row, col):
    """
    Where grid cell (row, col) of `trigger["grid"]` lands in an image of
    the given size, scaled proportionally from the trigger's reference
    frame. Shared by the matcher (reads the block) and by test/demo data
    generation (writes the block) so the geometry is defined exactly once.
    """
    scale_w = width / trigger["ref_dim"]
    scale_h = height / trigger["ref_dim"]
    row_start = int(round((trigger["ref_row_top"] + row) * scale_h))
    row_end = max(row_start + 1, int(round((trigger["ref_row_top"] + row + 1) * scale_h)))
    col_start = int(round((trigger["ref_col_top"] + col) * scale_w))
    col_end = max(col_start + 1, int(round((trigger["ref_col_top"] + col + 1) * scale_w)))
    return min(row_start, height), min(row_end, height), min(col_start, width), min(col_end, width)


def _match_fixed_grid_pattern(image_rgb, trigger):
    width, height = image_rgb.size
    arr = np.asarray(image_rgb)

    matches = 0
    total = 0
    for row, cells in enumerate(trigger["grid"]):
        for col, expected in enumerate(cells):
            if expected is None:
                continue  # masked position — not part of the signature
            total += 1
            row_start, row_end, col_start, col_end = _grid_cell_bounds(width, height, trigger, row, col)
            if row_start >= row_end or col_start >= col_end:
                continue
            block = arr[row_start:row_end, col_start:col_end, :3]
            mean_color = block.reshape(-1, block.shape[-1]).mean(axis=0)
            if np.all(np.abs(mean_color - expected) <= trigger["color_tolerance"]):
                matches += 1
    return matches / total if total else 0.0


_MATCHERS = {
    "solid_color_corner_patch": _match_solid_color_corner_patch,
    "fixed_grid_pattern": _match_fixed_grid_pattern,
}


def run_materialized_trigger_detector(samples):
    """
    samples: list of {"id": str, "image": PIL.Image (RGB)}
    Returns (report_dict, flagged_id_set).
    """
    flagged_ids = []

    for sample in samples:
        image = sample["image"]
        for trigger in TRIGGER_REGISTRY:
            matcher = _MATCHERS[trigger["type"]]
            match_ratio = matcher(image, trigger)
            if match_ratio >= trigger["match_ratio"]:
                flagged_ids.append(sample["id"])
                break  # one hit is enough to flag this sample

    report = {
        "name": "materialized_trigger_detector",
        "confidence": "high",
        "action": "auto_remove",
        "description": "Direct pixel-level match against known trigger signatures",
        "known_triggers_checked": [t["id"] for t in TRIGGER_REGISTRY],
        "flagged_count": len(flagged_ids),
        "flagged_sample_ids": flagged_ids,
    }
    return report, set(flagged_ids)
