"""
Shared model definition + preprocessing for the Dataset Cleaning &
Poison Detection service.

SmallCifarClassifier is the V1 feature extractor. It's a fast-start choice:
good enough to get the two-stage detection pipeline working end-to-end, but
it won't generalize well to real-world image domains outside a CIFAR-10-like
distribution. The next iteration should swap this for a general pretrained
backbone (e.g. ResNet) as the frozen feature extractor. That swap should
only touch this file — the rest of the pipeline (asset bundle format,
detectors, orchestration) is written against the functions below, not
against the network's internals.
"""
import torch
from torch import nn
from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
INPUT_SIZE = 32
NUM_CLASSES = 10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class SmallCifarClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_preprocess(input_size=INPUT_SIZE):
    # Buyers can upload images of any size/domain — resize to the fixed
    # size the classifier expects before anything else.
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def extract_features_and_logits(model, inputs):
    features = model.features(inputs)
    features = torch.flatten(features, 1)
    logits = model.classifier(features)
    return features, logits


def load_model(weights_path, device, num_classes=NUM_CLASSES):
    model = SmallCifarClassifier(num_classes=num_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
