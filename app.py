"""
app.py — Watermark Removal API
Combined Pipeline: Adaptive Gaussian Noise + MSJPEG-Y Attack
No ML models, no GPU, ~10MB total deployment
"""

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import os
import uuid

# ==========================================================
# App Setup
# ==========================================================

app = FastAPI(title="Watermark Removal")

os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==========================================================
# Attack Pipeline (no ML models needed)
# ==========================================================

def normalize_map(data):
    data = data.astype(np.float32)
    mn, mx = data.min(), data.max()
    if mx - mn < 1e-7:
        return np.zeros_like(data)
    return (data - mn) / (mx - mn)


def compute_importance_map(image):
    """
    Fast importance map using OpenCV operations only.
    Texture + gradient + variance via box filter trick.
    ~100x faster than scipy.generic_filter.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Fast texture: local std via box filter trick
    mean = cv2.blur(gray, (7, 7))
    sq_mean = cv2.blur(gray * gray, (7, 7))
    texture = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
    texture = normalize_map(texture)

    # Gradient: Sobel
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = normalize_map(cv2.magnitude(gx, gy))

    # Fast variance: same box filter trick
    variance = np.maximum(sq_mean - mean * mean, 0)
    variance = normalize_map(variance)

    imp = 0.4 * texture + 0.3 * gradient + 0.3 * variance
    return normalize_map(imp)


def adaptive_gaussian(image, importance, min_sigma=1.0, max_sigma=6.0):
    """
    Adaptive Gaussian noise — adds more noise in textured
    regions (high importance), preserves smooth areas and edges.
    """
    f = image.astype(np.float32)
    sigma = (min_sigma + importance * (max_sigma - min_sigma))[:, :, np.newaxis]
    noise = np.random.normal(0, 1, f.shape).astype(np.float32)
    return np.clip(f + noise * sigma, 0, 255).astype(np.uint8)


def msjpeg_y_attack(image, strength=0.8, jpeg_q=20, target_res=128):
    """
    Multi-Scale JPEG Y-channel attack.
    Works at 128px internal resolution, attacks only luminance.
    Cb/Cr channels untouched = zero colour shift.
    """
    h, w = image.shape[:2]
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = ycrcb[:, :, 0]

    Y_small = cv2.resize(Y, (target_res, target_res), interpolation=cv2.INTER_AREA)

    Y_uint8 = np.clip(Y_small, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode('.jpg', Y_uint8, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    Y_jpeg = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    diff_small = Y_jpeg - Y_small
    thr = np.percentile(np.abs(diff_small), 85)
    diff_small = np.clip(diff_small, -thr, thr)

    diff_full = cv2.resize(diff_small, (w, h), interpolation=cv2.INTER_LINEAR)

    ycrcb[:, :, 0] = np.clip(Y + diff_full * strength, 0, 255)
    result = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    return result


def _run_attack(image):
    """Core attack at working resolution."""
    imp = compute_importance_map(image)
    noisy = adaptive_gaussian(image, imp, min_sigma=1.0, max_sigma=6.0)
    attacked = msjpeg_y_attack(noisy, strength=0.8, jpeg_q=20, target_res=128)
    result = cv2.bilateralFilter(attacked, d=7, sigmaColor=20, sigmaSpace=20)
    return result


def combined_attack(image):
    """
    Full pipeline with resolution handling.
    Downscales large images to 512 for attack, then upscales back.
    Attack was validated at 512px — running at higher res weakens it.
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)

    if max_dim > 512:
        scale = 512 / max_dim
        small = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
        result_small = _run_attack(small)
        result = cv2.resize(result_small, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        result = _run_attack(image)

    return result


# ==========================================================
# Routes
# ==========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    output = combined_attack(image)

    uid = uuid.uuid4().hex[:8]
    filename = f"{uid}.png"
    path = os.path.join("outputs", filename)
    cv2.imwrite(path, output)

    return JSONResponse({
        "success": True,
        "download_url": f"/api/download/{filename}"
    })


@app.get("/api/download/{filename}")
async def download(filename: str):
    path = os.path.join("outputs", filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, media_type="image/png", filename="cleaned.png")
