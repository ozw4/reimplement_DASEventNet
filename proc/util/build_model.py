"""Utility to construct the CNN model."""

from torch import nn
from torchvision.models import resnet50


def build_model(input_channels: int = 1) -> nn.Module:
        """ResNet-50 encoder with custom first conv and sigmoid output."""
        model = resnet50(weights=None)
        # Adapt first conv layer to 1 channel
        model.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Replace final FC for binary classification
        model.fc = nn.Sequential(nn.Flatten(), nn.Linear(2048, 1))
        return model
