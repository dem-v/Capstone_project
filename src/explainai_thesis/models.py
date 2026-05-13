from __future__ import annotations

import torch
from torch import nn


class TinyCnn(nn.Module):
    """Small CNN used only for the synthetic smoke test."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(48, 2)

    @property
    def target_layer(self) -> nn.Module:
        return self.features[-2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
