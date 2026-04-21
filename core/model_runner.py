import threading
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import SimpleCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = Path(__file__).parent.parent / "weights" / "baseline.pt"

_model = None
_model_lock = threading.Lock()


def get_model() -> SimpleCNN:
    global _model
    if _model is None:
        with _model_lock:
            # double-checked locking: lock 획득 후 다시 확인
            if _model is None:
                m = SimpleCNN().to(DEVICE)
                m.load_state_dict(
                    torch.load(weights_path, map_location=DEVICE, weights_only=True)
                )
                m.eval()
                _model = m
    return _model


def run_inference(x):
    model = get_model()
    with torch.no_grad():
        logits = model(x.to(DEVICE))[0]
        probs = F.softmax(logits, dim=0).cpu().numpy().tolist()

    return {
        "pred": int(np.argmax(probs)),
        "conf": float(max(probs)),
        "probs": probs
    }
