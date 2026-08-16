"""
Offline, one-time asset preparation for the Dataset Cleaning & Poison
Detection listing.

This is what "the seller uploads a fully-trained model" means in practice
for V1: run this once, using the seller's own trusted reference data
(here: CIFAR-10's clean training set, standing in for that trusted data),
to produce the bundle that ships with the listing:

    server/assets/model_weights.pt   — SmallCifarClassifier state_dict
    server/assets/centroids.pt       — per-class feature centroids
    server/assets/manifest.json      — threshold + everything needed to
                                        reproduce preprocessing at run time

The buyer-facing pipeline (server/pipeline.py) only ever *reads* this
bundle — it never retrains or recomputes centroids from a buyer's upload.

Usage:
    python -m server.scripts.prepare_assets
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server.model import (  # noqa: E402
    CIFAR10_MEAN,
    CIFAR10_STD,
    INPUT_SIZE,
    NUM_CLASSES,
    SmallCifarClassifier,
    extract_features_and_logits,
)

ASSET_DIR = os.path.join(ROOT_DIR, "server", "assets")
DATA_DIR = os.path.join(ROOT_DIR, "server", ".data")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image, label, index


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_correct = total = 0
    for inputs, labels, _ in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(1, total), 100.0 * total_correct / max(1, total)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total = 0
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(1, total), 100.0 * total_correct / max(1, total)


def compute_centroids(model, loader, device, num_classes):
    sums = [None] * num_classes
    counts = [0] * num_classes
    model.eval()
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features, _ = extract_features_and_logits(model, inputs)
            for label in labels.unique():
                label_value = int(label.item())
                mask = labels == label
                label_sum = features[mask].sum(dim=0)
                sums[label_value] = label_sum if sums[label_value] is None else sums[label_value] + label_sum
                counts[label_value] += int(mask.sum().item())
    centroids = []
    for label_value in range(num_classes):
        if counts[label_value] == 0:
            raise ValueError(f"No reference samples found for class {label_value}")
        centroids.append(sums[label_value] / counts[label_value])
    return torch.stack(centroids, dim=0)


def score_batch(model, centroids, inputs, device):
    with torch.no_grad():
        features, logits = extract_features_and_logits(model, inputs)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_labels = probs.max(dim=1)
        distances = torch.cdist(features, centroids)
        pred_dist = distances[torch.arange(distances.size(0), device=device), pred_labels]
        return pred_dist + (1.0 - confidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ASSET_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomCrop(INPUT_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_base = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    eval_base = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=eval_transform)

    indices = list(range(len(train_base)))
    random.Random(args.seed).shuffle(indices)
    val_size = max(1, int(len(indices) * args.val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_loader = DataLoader(
        IndexedDataset(Subset(train_base, train_indices)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    train_eval_loader = DataLoader(
        IndexedDataset(Subset(eval_base, train_indices)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    val_loader = DataLoader(
        IndexedDataset(Subset(eval_base, val_indices)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = SmallCifarClassifier(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(
            f"[prepare_assets] epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}", flush=True,
        )

    centroids = compute_centroids(model, train_eval_loader, device, NUM_CLASSES)

    scores = []
    model.eval()
    with torch.no_grad():
        for inputs, _labels, _idx in val_loader:
            inputs = inputs.to(device)
            scores.extend(score_batch(model, centroids, inputs, device).detach().cpu().tolist())
    threshold = float(np.quantile(np.asarray(scores, dtype=np.float32), args.threshold_quantile))

    torch.save(model.state_dict(), os.path.join(ASSET_DIR, "model_weights.pt"))
    torch.save(centroids.detach().cpu(), os.path.join(ASSET_DIR, "centroids.pt"))

    manifest = {
        "architecture": "SmallCifarClassifier",
        "num_classes": NUM_CLASSES,
        "input_size": INPUT_SIZE,
        "mean": list(CIFAR10_MEAN),
        "std": list(CIFAR10_STD),
        "score_mode": "pred_dist_plus_uncertainty",
        "threshold_quantile": args.threshold_quantile,
        "threshold": threshold,
        "reference_train_samples": len(train_indices),
        "reference_val_samples": len(val_indices),
        "final_val_accuracy": val_acc,
    }
    with open(os.path.join(ASSET_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
