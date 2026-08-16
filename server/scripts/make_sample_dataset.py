"""
Builds server/samples/sample_dataset.zip — the dataset the "or click to
load a sample" button in the Run page sends to the backend, so the demo
works without the visitor needing to prepare their own .zip.

Mix: mostly clean CIFAR-10 test images, some with the registered red-square
trigger injected (stage 1 should catch these), and a few with unrelated
heavy pixel corruption standing in for a "novel"/unknown poisoning pattern
(stage 2 — blind_feature — should catch these; they don't match the known
trigger signature).

Usage:
    python -m server.scripts.make_sample_dataset
"""
import argparse
import io
import json
import os
import random
import sys
import zipfile

import numpy as np
from PIL import Image
from torchvision import datasets

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server.model import CIFAR10_CLASSES  # noqa: E402
from server.detectors.materialized_trigger import TRIGGER_REGISTRY, _patch_size  # noqa: E402

SAMPLES_DIR = os.path.join(ROOT_DIR, "server", "samples")
DATA_DIR = os.path.join(ROOT_DIR, "server", ".data")


def inject_red_square(image, trigger):
    arr = np.array(image).copy()
    height, width = arr.shape[0], arr.shape[1]
    patch_w, patch_h = _patch_size(width, height, trigger)
    arr[height - patch_h:height, width - patch_w:width] = trigger["color"]
    return Image.fromarray(arr)


def inject_noise_anomaly(image, rng):
    arr = np.array(image).astype(np.int16)
    noise = rng.integers(-120, 120, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--trigger-ratio", type=float, default=0.12)
    parser.add_argument("--anomaly-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

    by_class = {i: [] for i in range(10)}
    for idx in range(len(test_set)):
        image, label = test_set[idx]
        if len(by_class[label]) < args.per_class:
            by_class[label].append(image)
        if all(len(v) >= args.per_class for v in by_class.values()):
            break

    trigger = TRIGGER_REGISTRY[0]
    zip_path = os.path.join(SAMPLES_DIR, "sample_dataset.zip")
    manifest_rows = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, images in by_class.items():
            class_name = CIFAR10_CLASSES[label]
            for i, image in enumerate(images):
                kind = "clean"
                roll = py_rng.random()
                if roll < args.trigger_ratio:
                    image = inject_red_square(image, trigger)
                    kind = "triggered"
                elif roll < args.trigger_ratio + args.anomaly_ratio:
                    image = inject_noise_anomaly(image, np_rng)
                    kind = "anomaly"

                name = f"{class_name}/{class_name}_{i:03d}.png"
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                zf.writestr(name, buf.getvalue())
                manifest_rows.append({"id": name, "kind": kind})

    with open(os.path.join(SAMPLES_DIR, "sample_dataset_manifest.json"), "w") as f:
        json.dump(manifest_rows, f, indent=2)

    counts = {}
    for row in manifest_rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    print(f"Wrote {zip_path} ({len(manifest_rows)} images): {counts}")


if __name__ == "__main__":
    main()
