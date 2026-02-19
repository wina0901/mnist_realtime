#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py

드로잉 입력(캔버스 이미지)을 MNIST 입력 형태로 변환하는 전처리 모듈.

핵심:
- grayscale
- resize 28x28
- normalize to [0,1]
- (옵션) invert: 흰 배경/검정 글씨 -> MNIST 스타일(검정 배경/흰 글씨)처럼 뒤집기
- (옵션) center: 그린 부분의 bounding box를 찾아 중앙으로 이동

반환:
- x_tensor: torch.Tensor shape (1,1,28,28), float32
- x784: np.ndarray shape (784,), float32
- preview28: PIL.Image (L) 28x28, 전처리 결과 확인용
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class PreprocessConfig:
    invert: bool = True
    center: bool = True
    threshold: float = 0.05  # center용 픽셀 threshold (0~1)


def extract_pil_from_gradio(value: Any) -> Optional[Image.Image]:
    """
    Gradio ImageEditor / Image 입력은 보통 다음 형태 중 하나로 들어옴:
    - dict with "composite": PIL.Image or ndarray
    - PIL.Image
    - numpy array

    이 함수를 app.py에서 사용하면 입력 처리 깔끔해짐.
    """
    if value is None:
        return None

    if isinstance(value, Image.Image):
        return value

    if isinstance(value, dict):
        comp = value.get("composite", None)
        if comp is None:
            return None
        if isinstance(comp, Image.Image):
            return comp
        # numpy array
        try:
            return Image.fromarray(comp)
        except Exception:
            return None

    # numpy array
    try:
        return Image.fromarray(value)
    except Exception:
        return None


def pil_to_mnist_tensors(img: Image.Image, cfg: PreprocessConfig) -> Tuple[torch.Tensor, np.ndarray, Image.Image]:
    """
    PIL 이미지를 MNIST 입력으로 변환.
    """
    # 1) grayscale
    img = img.convert("L")

    # 2) resize to 28x28
    img = img.resize((28, 28), resample=Image.Resampling.LANCZOS)

    # 3) normalize [0,1]
    arr = np.asarray(img).astype(np.float32) / 255.0

    # 4) invert (흰 배경/검정 글씨인 경우 보통 True 추천)
    if cfg.invert:
        arr = 1.0 - arr

    # 5) center (도메인 갭 완화에 도움)
    if cfg.center:
        arr = center_28x28(arr, threshold=cfg.threshold)

    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)

    # preview image
    preview28 = Image.fromarray((arr * 255).astype(np.uint8), mode="L")

    # flatten 784
    x784 = arr.reshape(-1).astype(np.float32)

    # tensor (1,1,28,28)
    x_tensor = torch.tensor(arr[None, None, :, :], dtype=torch.float32)

    return x_tensor, x784, preview28


def center_28x28(arr28: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    28x28(0~1) 배열에서 그려진 부분의 bounding box를 찾아 중앙으로 이동.

    - threshold보다 큰 픽셀들을 '그려진 픽셀'로 간주
    - 아무것도 없으면 원본 반환
    """
    if arr28.shape != (28, 28):
        raise ValueError(f"arr28 must be shape (28,28), got {arr28.shape}")

    ys, xs = np.where(arr28 > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return arr28

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    crop = arr28[y0:y1 + 1, x0:x1 + 1]
    h, w = crop.shape

    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - h) // 2
    left = (28 - w) // 2
    canvas[top:top + h, left:left + w] = crop
    return canvas
