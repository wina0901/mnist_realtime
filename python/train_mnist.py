#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mnist.py
- mode=baseline: 공식 MNIST로 학습 후 가중치 저장
- mode=finetune: baseline 가중치 로드 + data/custom 폴더(사용자 피드백 데이터)로 추가 학습

커스텀 데이터 폴더 구조(권장):
data/custom/
  0/ 1/ ... 9/
    *.png  (흑백/컬러 무관, 28x28이 아니어도 됨. 학습 시 28x28로 리사이즈)

예시:
  python python/train_mnist.py --mode baseline --epochs 5 --out python/weights/baseline.pt
  python python/train_mnist.py --mode finetune --ckpt python/weights/baseline.pt --custom_dir data/custom --epochs 3 --out python/weights/finetuned.pt
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
from PIL import Image


# -----------------------------
# Model
# -----------------------------
class SimpleCNN(nn.Module):
    """
    가벼운 MNIST CNN (빠르고 안정적인 baseline)
    입력: (N,1,28,28)
    출력: (N,10) logits
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 28x28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 28x28
        self.pool = nn.MaxPool2d(2, 2)                # 14x14
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))          # (N,16,14,14)
        x = F.relu(self.conv2(x))                     # (N,32,14,14)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -----------------------------
# Custom Dataset
# -----------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class CustomConfig:
    invert: bool = True  # 흰 배경/검정 글씨로 저장된 경우 보통 True 추천(=1-arr)


class CustomDigitsFolder(Dataset):
    """
    data/custom/{0..9}/*.png 구조를 읽는 커스텀 데이터셋.
    """
    def __init__(self, root_dir: str, cfg: CustomConfig) -> None:
        self.root_dir = root_dir
        self.cfg = cfg
        self.samples: List[Tuple[str, int]] = []

        if not os.path.isdir(root_dir):
            return

        for label in range(10):
            label_dir = os.path.join(root_dir, str(label))
            if not os.path.isdir(label_dir):
                continue
            for fn in os.listdir(label_dir):
                if fn.lower().endswith(IMG_EXTS):
                    self.samples.append((os.path.join(label_dir, fn), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("L").resize((28, 28), resample=Image.Resampling.LANCZOS)
        arr = (np.asarray(img).astype(np.float32) / 255.0)  # 0~1
        if self.cfg.invert:
            arr = 1.0 - arr
        arr = np.clip(arr, 0.0, 1.0)

        x = torch.tensor(arr[None, :, :], dtype=torch.float32)  # (1,28,28)
        return x, label


# -----------------------------
# Train / Eval
# -----------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / max(len(loader.dataset), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "finetune"], default="baseline")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--data_dir", type=str, default="./data", help="MNIST 다운로드/캐시 디렉토리")
    p.add_argument("--custom_dir", type=str, default="./data/custom", help="커스텀 드로잉 데이터 폴더")
    p.add_argument("--invert_custom", action="store_true", help="커스텀 이미지가 흰배경/검정글씨면 켜기(대부분 ON 권장)")

    p.add_argument("--ckpt", type=str, default="./python/weights/baseline.pt", help="finetune 시작 가중치 경로")
    p.add_argument("--out", type=str, default="./python/weights/baseline.pt", help="저장 경로 (.pt)")
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # MNIST
    tfm = transforms.Compose([transforms.ToTensor()])  # (1,28,28), 0~1
    mnist_train = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=tfm)
    mnist_test = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=tfm)

    # Dataset 구성
    if args.mode == "baseline":
        train_ds = mnist_train
        print("[INFO] mode=baseline (MNIST only)")
    else:
        custom_cfg = CustomConfig(invert=True if args.invert_custom else True)  # 기본 True
        custom_ds = CustomDigitsFolder(args.custom_dir, cfg=custom_cfg)
        print(f"[INFO] mode=finetune (MNIST + custom). custom samples = {len(custom_ds)}")

        if len(custom_ds) < 20:
            print("[WARN] 커스텀 샘플이 매우 적습니다. (20개 미만) 과적합 가능성이 큽니다.")

        train_ds = ConcatDataset([mnist_train, custom_ds])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        mnist_test,
        batch_size=512,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = SimpleCNN().to(device)

    # finetune이면 ckpt 로드
    if args.mode == "finetune":
        if os.path.isfile(args.ckpt):
            print(f"[INFO] loading ckpt: {args.ckpt}")
            state = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(state)
        else:
            print(f"[WARN] ckpt not found: {args.ckpt} (랜덤 init으로 finetune 진행)")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optim, device)
        acc = evaluate(model, test_loader, device)
        print(f"[EPOCH {epoch:02d}/{args.epochs}] loss={loss:.4f}  test_acc={acc*100:.2f}%")

    torch.save(model.state_dict(), args.out)
    print(f"[DONE] saved weights -> {args.out}")


if __name__ == "__main__":
    main()
