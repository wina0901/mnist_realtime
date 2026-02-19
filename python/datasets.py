#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets.py

커스텀 손글씨 데이터 로더.
폴더 구조:

data/custom/
  ├─ 0/
  │    ├─ xxx.png
  ├─ 1/
  │    ├─ yyy.png
  ...
  └─ 9/

각 폴더 이름이 곧 라벨(0~9).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class CustomDigitsDataset(Dataset):
    """
    data/custom/{0..9}/*.png 구조를 읽는 Dataset

    Args:
        root_dir: 커스텀 데이터 루트 경로 (예: data/custom)
        invert: 흰 배경/검정 글씨로 저장된 경우 True 권장
        resize: 항상 28x28로 리사이즈 (MNIST 입력 맞춤)
    """

    def __init__(self, root_dir: str, invert: bool = True, resize: bool = True) -> None:
        self.root_dir = root_dir
        self.invert = invert
        self.resize = resize
        self.samples: List[Tuple[str, int]] = []

        if not os.path.isdir(root_dir):
            print(f"[WARN] Custom dataset directory not found: {root_dir}")
            return

        for label in range(10):
            label_dir = os.path.join(root_dir, str(label))
            if not os.path.isdir(label_dir):
                continue

            for fname in os.listdir(label_dir):
                if fname.lower().endswith(IMG_EXTS):
                    fpath = os.path.join(label_dir, fname)
                    self.samples.append((fpath, label))

        print(f"[INFO] Loaded custom dataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("L")  # grayscale

        if self.resize:
            img = img.resize((28, 28), resample=Image.Resampling.LANCZOS)

        arr = np.asarray(img).astype(np.float32) / 255.0  # 0~1

        if self.invert:
            arr = 1.0 - arr

        arr = np.clip(arr, 0.0, 1.0)

        # (1, 28, 28) 형태로 변환
        tensor = torch.tensor(arr[None, :, :], dtype=torch.float32)

        return tensor, label


def count_custom_samples(root_dir: str) -> int:
    """
    커스텀 데이터 총 샘플 개수 반환 (UI 표시용)
    """
    total = 0
    if not os.path.isdir(root_dir):
        return 0

    for label in range(10):
        label_dir = os.path.join(root_dir, str(label))
        if not os.path.isdir(label_dir):
            continue

        for fname in os.listdir(label_dir):
            if fname.lower().endswith(IMG_EXTS):
                total += 1

    return total
