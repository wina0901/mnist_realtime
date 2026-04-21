from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
 
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.pool  = nn.MaxPool2d(2, 2)                           # 각 conv 뒤에서 절반으로
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (N, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (N, 32,  7,  7)  ← pool 추가
        x = x.view(x.size(0), -1)              # (N, 1568)
        x = F.relu(self.fc1(x))                # (N, 128)
        logits = self.fc2(x)                   # (N, 10)
        return logits


class SimpleMLP(nn.Module):

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)              # (N, 784)
        x = F.relu(self.fc1(x))                # (N, hidden)
        logits = self.fc2(x)                   # (N, 10)
        return logits


def build_model(name: str = "cnn", **kwargs) -> nn.Module:

    name = name.lower().strip()
    if name in ("cnn", "simplecnn"):
        return SimpleCNN()
    if name in ("mlp", "simplemlp"):
        hidden = int(kwargs.get("hidden", 256))
        return SimpleMLP(hidden=hidden)
    raise ValueError(f"Unknown model name: {name}")
