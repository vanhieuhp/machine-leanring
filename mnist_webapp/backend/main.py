import io
import base64
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from torchvision import transforms

from model import LeNet

app = FastAPI(title="MNIST Digit Recognizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "lenet_mnist.pt"

model = LeNet()
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print(f"Model loaded from {MODEL_PATH}")
else:
    print("WARNING: lenet_mnist.pt not found — run train.py first")
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def preprocess(image: Image.Image) -> torch.Tensor:
    # Convert to grayscale
    img = image.convert("L")

    # Crop to bounding box of drawn content with padding, then resize to 20x20
    # and center-pad to 28x28 — matches how MNIST digits are stored
    arr = np.array(img)
    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        pad = max(int((rmax - rmin) * 0.15), int((cmax - cmin) * 0.15), 2)
        rmin = max(rmin - pad, 0)
        rmax = min(rmax + pad, arr.shape[0] - 1)
        cmin = max(cmin - pad, 0)
        cmax = min(cmax + pad, arr.shape[1] - 1)

        img = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Resize to 20x20, then paste on 28x28 black canvas centered
    img = img.resize((20, 20), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, (4, 4))
    img = canvas

    return transform(img).unsqueeze(0)  # (1, 1, 28, 28)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Cannot decode image")

    tensor = preprocess(image)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().tolist()

    digit = int(np.argmax(probs))
    confidence = float(probs[digit])

    return JSONResponse({
        "digit": digit,
        "confidence": round(confidence * 100, 2),
        "probabilities": [round(p * 100, 2) for p in probs],
    })


@app.post("/predict_base64")
async def predict_base64(payload: dict):
    data_url = payload.get("image", "")
    if not data_url:
        raise HTTPException(400, "Missing image field")

    # Strip "data:image/png;base64," prefix if present
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        image = Image.open(io.BytesIO(base64.b64decode(data_url)))
    except Exception:
        raise HTTPException(400, "Cannot decode base64 image")

    tensor = preprocess(image)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().tolist()

    digit = int(np.argmax(probs))
    confidence = float(probs[digit])

    return JSONResponse({
        "digit": digit,
        "confidence": round(confidence * 100, 2),
        "probabilities": [round(p * 100, 2) for p in probs],
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
