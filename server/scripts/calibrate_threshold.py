"""
Recompute the blind_feature anomaly threshold at a different quantile
WITHOUT retraining — reuses the already-trained model_weights.pt +
centroids.pt in server/assets/, rebuilds the same held-out validation
split used by prepare_assets.py (same seed, same val_ratio), and reports
the threshold for one or more candidate quantiles.

Pass --apply <quantile> to write manifest.json with that threshold. With
no --apply, this only prints candidates — nothing on disk changes.

Usage:
    python -m server.scripts.calibrate_threshold --quantiles 0.95 0.97
    python -m server.scripts.calibrate_threshold --apply 0.95
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server.model import CIFAR10_MEAN, CIFAR10_STD, INPUT_SIZE, load_model  # noqa: E402
from server.scripts.prepare_assets import IndexedDataset, score_batch  # noqa: E402

ASSET_DIR = os.path.join(ROOT_DIR, "server", "assets")
DATA_DIR = os.path.join(ROOT_DIR, "server", ".data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.95, 0.97])
    parser.add_argument("--apply", type=float, default=None, help="Write this quantile's threshold to manifest.json")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(ASSET_DIR, "manifest.json")) as f:
        manifest = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(os.path.join(ASSET_DIR, "model_weights.pt"), device)
    centroids = torch.load(os.path.join(ASSET_DIR, "centroids.pt"), map_location=device)

    eval_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_base = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=eval_transform)

    # Reproduce the exact same train/val split prepare_assets.py used.
    indices = list(range(len(eval_base)))
    random.Random(args.seed).shuffle(indices)
    val_size = max(1, int(len(indices) * args.val_ratio))
    val_indices = indices[:val_size]

    val_loader = DataLoader(
        IndexedDataset(Subset(eval_base, val_indices)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    scores = []
    model.eval()
    with torch.no_grad():
        for inputs, _labels, _idx in val_loader:
            inputs = inputs.to(device)
            scores.extend(score_batch(model, centroids, inputs, device).detach().cpu().tolist())
    scores = np.asarray(scores, dtype=np.float32)

    print(f"validation samples: {len(scores)}  (matches manifest reference_val_samples: {manifest['reference_val_samples']})")
    results = {}
    for q in args.quantiles:
        threshold = float(np.quantile(scores, q))
        results[q] = threshold
        print(f"quantile={q:.2f}  threshold={threshold:.4f}")

    if args.apply is not None:
        if args.apply not in results:
            threshold = float(np.quantile(scores, args.apply))
        else:
            threshold = results[args.apply]
        manifest["threshold_quantile"] = args.apply
        manifest["threshold"] = threshold
        with open(os.path.join(ASSET_DIR, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nApplied: threshold_quantile={args.apply}, threshold={threshold:.4f} -> {os.path.join(ASSET_DIR, 'manifest.json')}")


if __name__ == "__main__":
    main()
