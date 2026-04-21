import redis
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import SimpleCNN
from core.preprocess import preprocess   

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

QUEUE_NAME = "mnist_queue"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = Path(__file__).parent / "weights" / "baseline.pt"

model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
model.eval()


print("🚀 Worker started...")

while True:
    # brpop으로 블로킹 대기 (CPU 낭비 없음)
    item = r.brpop(QUEUE_NAME, timeout=1)

    if item is None:
        continue

    _, job_data = item

    job = json.loads(job_data)
    job_id = job["id"]

    try:
        image_bytes = bytes.fromhex(job["image"])
        img = Image.open(io.BytesIO(image_bytes))

        x = preprocess(img)
        if x is None:
            result = {"error": "empty image"}
        else:
            with torch.no_grad():
                logits = model(x.to(DEVICE))[0]
                probs = F.softmax(logits, dim=0).cpu().numpy().tolist()

            result = {
                "pred": int(np.argmax(probs)),
                "conf": float(max(probs)),
                "probs": probs
            }

    except Exception as e:
        result = {"error": str(e)}

    # TTL 300초 설정 (메모리 누수 방지)
    r.set(job_id, json.dumps(result), ex=300)

    print(f"✅ done: {job_id}")
