#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feedback_log.py

사용자 피드백(맞음/틀림)을 jsonl로 기록하는 유틸.

권장 로그 파일:
  data/custom/feedback.jsonl

각 줄은 JSON 1개 (json lines)
- timestamp_ms
- model_choice (baseline/finetuned 등)
- predicted (int)
- confidence (float)
- user_feedback ("correct" | "wrong")
- true_label (int | null)  # wrong일 때만 값 존재
- saved_path (str | null)  # 샘플 이미지 저장한 경우 경로
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class FeedbackRecord:
    timestamp_ms: int
    model_choice: str
    predicted: int
    confidence: float
    user_feedback: str                 # "correct" or "wrong"
    true_label: Optional[int] = None
    saved_path: Optional[str] = None


def now_ms() -> int:
    return int(time.time() * 1000)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_feedback(log_path: str, record: FeedbackRecord) -> None:
    """
    JSONL로 한 줄 append
    """
    ensure_parent_dir(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_feedback_tail(log_path: str, n: int = 20):
    """
    최근 n줄을 읽어 반환 (디버깅/표시용)
    """
    if not os.path.isfile(log_path):
        return []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[-n:]
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
