#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py
MNIST 실시간 드로잉 인식 + 피드백 루프(맞음/틀림) + 커스텀 저장 + 파인튜닝 버튼(옵션)

필요 파일/모듈:
- model.py        (SimpleCNN)
- preprocess.py   (Gradio 입력 -> 28x28 -> tensor)
- datasets.py     (커스텀 데이터 카운트)
- feedback_log.py (피드백 jsonl 기록)

가중치 경로(기본):
- python/weights/baseline.pt
- python/weights/finetuned.pt

커스텀 데이터 저장 경로(기본):
- data/custom/{0..9}/*.png
- data/custom/feedback.jsonl

실행:
  python python/app.py
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image

from model import SimpleCNN
from preprocess import PreprocessConfig, extract_pil_from_gradio, pil_to_mnist_tensors
from datasets import count_custom_samples
from feedback_log import FeedbackRecord, append_feedback, now_ms


# -----------------------------
# Paths / Settings
# -----------------------------
WEIGHTS_DIR = "./python/weights"
BASELINE_PATH = os.path.join(WEIGHTS_DIR, "baseline.pt")
FINETUNED_PATH = os.path.join(WEIGHTS_DIR, "finetuned.pt")

CUSTOM_ROOT = "./data/custom"
FEEDBACK_LOG_PATH = os.path.join(CUSTOM_ROOT, "feedback.jsonl")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
for d in range(10):
    os.makedirs(os.path.join(CUSTOM_ROOT, str(d)), exist_ok=True)


# -----------------------------
# Model utils
# -----------------------------
def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run_inference(img_pil: Image.Image, model_choice: str, cfg: PreprocessConfig) -> Tuple[int, float, List[float], Image.Image]:
    """
    Returns:
      pred (int), confidence (float), probs (list[10]), preview28 (PIL Image 28x28)
    """
    device = _device()
    x_tensor, x784, preview28 = pil_to_mnist_tensors(img_pil, cfg)
    x_tensor = x_tensor.to(device)

    model = SimpleCNN().to(device)

    wpath = BASELINE_PATH if model_choice == "baseline" else FINETUNED_PATH
    if os.path.isfile(wpath):
        model.load_state_dict(torch.load(wpath, map_location=device))
    else:
        # 가중치가 없으면 랜덤 init으로라도 동작하게(사용자 경험)
        print(f"[WARN] weights not found: {wpath} (random init)")

    model.eval()
    logits = model(x_tensor)[0].float().cpu()
    probs = F.softmax(logits, dim=0).numpy().astype(float).tolist()
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred, conf, probs, preview28


def probs_to_table(probs: List[float]) -> List[Dict[str, float]]:
    return [{"digit": i, "prob": float(probs[i])} for i in range(10)]


# -----------------------------
# Custom sample saving
# -----------------------------
def save_preprocessed_28(preview28: Image.Image, label: int) -> str:
    import time
    fname = f"{int(time.time() * 1000)}.png"
    out_dir = os.path.join(CUSTOM_ROOT, str(int(label)))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    preview28.save(out_path)
    return out_path


# -----------------------------
# UI handlers
# -----------------------------
def on_predict(editor_value: Any, model_choice: str, invert: bool, center: bool):
    img = extract_pil_from_gradio(editor_value)
    if img is None:
        # outputs: text, preview, probs_table, state_pred, state_conf, state_probs, state_preview28
        empty_table = probs_to_table([0.0] * 10)
        return "입력이 없어요. 캔버스에 숫자를 그려주세요.", None, empty_table, -1, 0.0, [0.0]*10, None

    cfg = PreprocessConfig(invert=invert, center=center)
    pred, conf, probs, preview28 = run_inference(img, model_choice, cfg)

    msg = f"예측: {pred} (confidence={conf:.3f})"
    table = probs_to_table(probs)
    return msg, preview28, table, pred, conf, probs, preview28


def on_mark_correct(model_choice: str, state_pred: int, state_conf: float):
    if state_pred is None or int(state_pred) < 0:
        return "먼저 예측을 해주세요. (✅/❌는 예측 이후에 가능)"

    rec = FeedbackRecord(
        timestamp_ms=now_ms(),
        model_choice=model_choice,
        predicted=int(state_pred),
        confidence=float(state_conf),
        user_feedback="correct",
        true_label=int(state_pred),
        saved_path=None,
    )
    append_feedback(FEEDBACK_LOG_PATH, rec)
    return "✅ 기록 완료: 맞아요 (로그 저장됨)"


def on_show_wrong_ui():
    # wrong 섹션 보이게
    return gr.update(visible=True), "❌ 틀려요: 정답 라벨을 선택하고 '정답으로 저장'을 눌러주세요."


def on_save_wrong(model_choice: str,
                  invert: bool,
                  center: bool,
                  true_label: int,
                  state_pred: int,
                  state_conf: float,
                  state_preview28: Any):
    """
    state_preview28: 예측 시 만들어둔 28x28 PIL 이미지
    """
    if state_pred is None or int(state_pred) < 0:
        return "먼저 예측을 해주세요.", count_custom_samples(CUSTOM_ROOT)

    if state_preview28 is None:
        return "전처리 이미지가 없어요. 다시 예측해 주세요.", count_custom_samples(CUSTOM_ROOT)

    # 저장
    if isinstance(state_preview28, Image.Image):
        preview28 = state_preview28
    else:
        try:
            preview28 = Image.fromarray(state_preview28)
        except Exception:
            return "전처리 이미지를 처리하지 못했어요. 다시 예측해 주세요.", count_custom_samples(CUSTOM_ROOT)

    saved_path = save_preprocessed_28(preview28, int(true_label))

    # 로그
    rec = FeedbackRecord(
        timestamp_ms=now_ms(),
        model_choice=model_choice,
        predicted=int(state_pred),
        confidence=float(state_conf),
        user_feedback="wrong",
        true_label=int(true_label),
        saved_path=saved_path,
    )
    append_feedback(FEEDBACK_LOG_PATH, rec)

    # wrong 섹션 숨기기 + 카운트 업데이트
    return f"❌ 저장 완료: 정답={int(true_label)} / saved={saved_path}", count_custom_samples(CUSTOM_ROOT)


def on_refresh_custom_count():
    return count_custom_samples(CUSTOM_ROOT)


def on_run_finetune(epochs: int, lr: float, batch_size: int):
    """
    간단하게 subprocess로 train_mnist.py finetune 실행.
    학습이 길면 UI가 잠깐 멈출 수 있으니 epochs는 작게 추천.
    """
    # baseline 가중치가 없으면 안내
    if not os.path.isfile(BASELINE_PATH):
        return "baseline.pt가 없습니다. 먼저 baseline 학습을 실행하세요:\npython python/train_mnist.py --mode baseline --out python/weights/baseline.pt"

    n_custom = count_custom_samples(CUSTOM_ROOT)
    if n_custom < 20:
        # 너무 적으면 경고
        warn = f"[WARN] 커스텀 샘플이 {n_custom}개로 매우 적습니다. 과적합될 수 있어요.\n그래도 진행합니다...\n\n"
    else:
        warn = ""

    cmd = [
        "python", "python/train_mnist.py",
        "--mode", "finetune",
        "--ckpt", BASELINE_PATH,
        "--custom_dir", CUSTOM_ROOT,
        "--epochs", str(int(epochs)),
        "--lr", str(float(lr)),
        "--batch_size", str(int(batch_size)),
        "--out", FINETUNED_PATH,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode != 0:
            return warn + "파인튜닝 실패 (로그 확인):\n" + out[-4000:]
        return warn + "✅ 파인튜닝 완료! finetuned.pt가 업데이트되었습니다.\n\n" + out[-4000:]
    except Exception as e:
        return warn + f"파인튜닝 실행 중 오류: {e}"


# -----------------------------
# UI
# -----------------------------
def build_ui():
    with gr.Blocks(title="MNIST Real-time Feedback (PyTorch + Gradio)") as demo:
        gr.Markdown(
            """
# MNIST 실시간 손글씨 숫자 인식 + 사용자 피드백 기반 파인튜닝
- **baseline**: 공식 MNIST로 학습한 모델
- **finetuned**: 사용자 드로잉(커스텀 데이터)로 추가 학습한 모델

흐름:
1) 숫자 그리기 → 2) 예측 → 3) ✅맞아요 / ❌틀려요 피드백 → 4) 샘플 축적 → 5) 파인튜닝 → 6) 개선 체감
"""
        )

        # 상태(예측 결과를 맞음/틀림 처리에 재사용)
        state_pred = gr.State(-1)
        state_conf = gr.State(0.0)
        state_probs = gr.State([0.0]*10)
        state_preview28 = gr.State(None)

        with gr.Row():
            editor = gr.ImageEditor(
                label="여기에 0~9 숫자를 그리세요 (마우스/터치)",
                height=280,
                width=280,
            )

            with gr.Column():
                model_choice = gr.Radio(
                    choices=["baseline", "finetuned"],
                    value="baseline",
                    label="모델 선택"
                )
                invert = gr.Checkbox(value=True, label="Invert (흰 배경/검정 글씨면 ON 추천)")
                center = gr.Checkbox(value=True, label="Centering (ON 추천)")

                predict_btn = gr.Button("예측하기")
                pred_text = gr.Textbox(label="예측 결과", lines=1)

                preview = gr.Image(label="전처리된 28x28", height=140, width=140)

                probs_plot = gr.BarPlot(
                    label="0~9 확률(신뢰도)",
                    x="digit",
                    y="prob",
                    title="Probabilities",
                    height=260,
                )

        with gr.Row():
            correct_btn = gr.Button("✅ 맞아요")
            wrong_btn = gr.Button("❌ 틀려요")
            feedback_msg = gr.Textbox(label="피드백 로그", lines=2)

        # 틀려요 섹션(기본 숨김)
        with gr.Row(visible=False) as wrong_section:
            true_label = gr.Slider(0, 9, value=0, step=1, label="정답 라벨(0~9)")
            save_wrong_btn = gr.Button("정답으로 저장 (커스텀 데이터 추가)")

        with gr.Row():
            custom_count = gr.Number(value=count_custom_samples(CUSTOM_ROOT), label="커스텀 샘플 개수", precision=0)
            refresh_btn = gr.Button("샘플 개수 새로고침")

        with gr.Accordion("파인튜닝 (커스텀 데이터로 추가 학습)", open=False):
            gr.Markdown(
                """
- 커스텀 샘플이 어느 정도(예: 50~200개) 쌓이면 실행하는 걸 추천합니다.
- epochs는 작게(2~5) 돌려서 빠르게 '개선 체감'을 만드는 게 데모에 좋아요.
"""
            )
            ft_epochs = gr.Slider(1, 10, value=3, step=1, label="epochs")
            ft_lr = gr.Number(value=1e-3, label="learning rate")
            ft_bs = gr.Slider(16, 256, value=128, step=16, label="batch size")
            finetune_btn = gr.Button("파인튜닝 실행")
            finetune_log = gr.Textbox(label="파인튜닝 로그", lines=12)

        # wiring
        predict_btn.click(
            fn=on_predict,
            inputs=[editor, model_choice, invert, center],
            outputs=[pred_text, preview, probs_plot, state_pred, state_conf, state_probs, state_preview28],
        ).then(
            fn=lambda probs_table: probs_table,
            inputs=[probs_plot],
            outputs=[probs_plot],
        )

        correct_btn.click(
            fn=on_mark_correct,
            inputs=[model_choice, state_pred, state_conf],
            outputs=[feedback_msg],
        )

        wrong_btn.click(
            fn=on_show_wrong_ui,
            inputs=[],
            outputs=[wrong_section, feedback_msg],
        )

        save_wrong_btn.click(
            fn=on_save_wrong,
            inputs=[model_choice, invert, center, true_label, state_pred, state_conf, state_preview28],
            outputs=[feedback_msg, custom_count],
        ).then(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[wrong_section],
        )

        refresh_btn.click(
            fn=on_refresh_custom_count,
            inputs=[],
            outputs=[custom_count],
        )

        finetune_btn.click(
            fn=on_run_finetune,
            inputs=[ft_epochs, ft_lr, ft_bs],
            outputs=[finetune_log],
        )

        gr.Markdown(
            """
## 학습 명령(수동)
```bash
# baseline
python python/train_mnist.py --mode baseline --epochs 5 --out python/weights/baseline.pt

# finetune (커스텀 데이터가 쌓인 후)
python python/train_mnist.py --mode finetune --ckpt python/weights/baseline.pt --custom_dir data/custom --epochs 3 --out python/weights/finetuned.pt
```
"""
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
