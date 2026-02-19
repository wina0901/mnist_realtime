#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py
MNIST 실시간 손글씨 인식용 모델 정의.

- 기본 제공: SimpleCNN (권장)
- 옵션: SimpleMLP (가중치 직접 곱(Wx+b) 구조를 강조하고 싶을 때 유용)

입력 규격:
  x: torch.Tensor shape (N, 1, 28, 28), 값 범위 [0,1] 권장

출력 규격:
  logits: torch.Tensor shape (N, 10)
  -> softmax는 추론 단계(app.py)에서 적용
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    가벼운 CNN 모델 (MNIST baseline으로 충분히 좋고, 실시간 데모에 적합)
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                            # 28x28 -> 14x14

        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (N,16,14,14)
        x = F.relu(self.conv2(x))              # (N,32,14,14)
        x = x.view(x.size(0), -1)              # (N, 32*14*14)
        x = F.relu(self.fc1(x))                # (N,128)
        logits = self.fc2(x)                   # (N,10)
        return logits


class SimpleMLP(nn.Module):
    """
    완전연결층 기반 MLP (원리 설명/가중치 내보내기/직접 곱 구현 강조에 좋음)

    구조:
      flatten(784) -> Linear(784, hidden) -> ReLU -> Linear(hidden, 10)

    주의:
      CNN보다 성능이 약간 떨어질 수 있고, 입력 전처리에 더 민감할 수 있음.
    """
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)              # (N,784)
        x = F.relu(self.fc1(x))                # (N,hidden)
        logits = self.fc2(x)                   # (N,10)
        return logits


def build_model(name: str = "cnn", **kwargs) -> nn.Module:
    """
    name:
      - "cnn": SimpleCNN
      - "mlp": SimpleMLP

    예:
      model = build_model("cnn")
      model = build_model("mlp", hidden=256)
    """
    name = name.lower().strip()
    if name in ("cnn", "simplecnn"):
        return SimpleCNN()
    if name in ("mlp", "simplemlp"):
        hidden = int(kwargs.get("hidden", 256))
        return SimpleMLP(hidden=hidden)
    raise ValueError(f"Unknown model name: {name}")
