import requests
import io

API_URL = "http://localhost:8000"


def call_api(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = requests.post(
        f"{API_URL}/predict",
        files={"file": ("image.png", buf, "image/png")}
    )
    res.raise_for_status()
    return res.json()
