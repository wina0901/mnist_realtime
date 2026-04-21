from core.model_runner import run_inference
from core.preprocess import preprocess

def predict(img):
    x = preprocess(img)
    if x is None:
        return {"error": "empty image"}

    return run_inference(x)