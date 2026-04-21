import numpy as np
import torch
from PIL import Image


def preprocess(img: Image.Image):
    rgba = np.array(img.convert("RGBA"))
    alpha = rgba[..., 3] / 255.0

    # 투명도 없는 이미지(흰 배경 검은 글씨, JPEG 등) 폴백 처리
    if alpha.max() < 0.5:
        gray = np.array(img.convert("L")) / 255.0
        mask = (1.0 - gray) > 0.15
    else:
        mask = alpha > 0.1

    if mask.max() < 0.05 or mask.sum() < 10:
        return None

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    if alpha.max() < 0.5:
        intensity = 1.0 - np.array(img.convert("L")) / 255.0
    else:
        intensity = alpha

    crop = intensity[y0:y1 + 1, x0:x1 + 1]

    size = max(crop.shape)
    sq = np.zeros((size, size))

    oy = (size - crop.shape[0]) // 2
    ox = (size - crop.shape[1]) // 2
    sq[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop

    sq_img = Image.fromarray((sq * 255).astype(np.uint8))
    resized = sq_img.resize((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(resized, (4, 4))

    arr = np.array(canvas) / 255.0
    return torch.tensor(arr[None, None, :, :], dtype=torch.float32)
