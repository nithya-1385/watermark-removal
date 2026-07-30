"""
app.py — Watermark Removal API
Matches Watermark_final.ipynb pipeline:
  Analysis → Adaptive Noise (Gaussian/Speckle/SaltPepper) → MSJPEG-Y → Bilateral
Runs all 3 noise variants, returns the best quality output.
No ML models, no GPU.
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
# Config (matches config.py in notebook)
# ==========================================================

TEXTURE_WEIGHT = 0.40
GRADIENT_WEIGHT = 0.30
VARIANCE_WEIGHT = 0.30

MIN_SIGMA = 1.0
MAX_SIGMA = 6.0

JPEG_QUALITY = 20
TARGET_RESOLUTION = 128
MSJPEG_STRENGTH = 1.0

BILATERAL_DIAMETER = 7
BILATERAL_SIGMA_COLOR = 20
BILATERAL_SIGMA_SPACE = 20


# ==========================================================
# Utils (matches utils.py in notebook)
# ==========================================================

def normalize_map(data):
    data = data.astype(np.float32)
    mn, mx = data.min(), data.max()
    if mx - mn < 1e-8:
        return np.zeros_like(data)
    return (data - mn) / (mx - mn)


# ==========================================================
# Analysis (matches analysis.py — fast cv2.blur version)
# ==========================================================

def compute_texture_map(image, window_size=7):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (window_size, window_size))
    sq_mean = cv2.blur(gray * gray, (window_size, window_size))
    texture = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
    return normalize_map(texture)


def compute_gradient_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return normalize_map(cv2.magnitude(gx, gy))


def compute_variance_map(image, window_size=7):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (window_size, window_size))
    sq_mean = cv2.blur(gray * gray, (window_size, window_size))
    variance = np.maximum(sq_mean - mean * mean, 0)
    return normalize_map(variance)


def analyze_image(image):
    texture = compute_texture_map(image)
    gradient = compute_gradient_map(image)
    variance = compute_variance_map(image)
    importance = normalize_map(
        TEXTURE_WEIGHT * texture +
        GRADIENT_WEIGHT * gradient +
        VARIANCE_WEIGHT * variance
    )
    return {
        "texture": texture,
        "gradient": gradient,
        "variance": variance,
        "importance": importance,
    }


# ==========================================================
# Adaptive Noise (matches adaptive_noise.py)
# ==========================================================

def compute_sigma_map(importance_map):
    return MIN_SIGMA + importance_map * (MAX_SIGMA - MIN_SIGMA)


def apply_adaptive_gaussian(image, importance_map):
    image = image.astype(np.float32)
    sigma = compute_sigma_map(importance_map)[:, :, np.newaxis]
    noise = np.random.normal(0, 1, image.shape).astype(np.float32)
    noisy = image + noise * sigma
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_adaptive_speckle(image, importance_map):
    image = image.astype(np.float32)
    sigma = compute_sigma_map(importance_map)[:, :, np.newaxis]
    noise = np.random.normal(0, 1, image.shape).astype(np.float32)
    noisy = image + image * noise * sigma / 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_adaptive_salt_pepper(image, importance_map):
    noisy = image.copy()
    probability = 0.01 + importance_map * 0.03
    random_map = np.random.rand(image.shape[0], image.shape[1])
    noisy[random_map > (1 - probability / 2)] = 255
    noisy[random_map < (probability / 2)] = 0
    return noisy


# ==========================================================
# MSJPEG-Y Attack (Nitsie's module)
# ==========================================================

def msjpeg_y_attack(image, strength=MSJPEG_STRENGTH, jpeg_q=JPEG_QUALITY,
                    target_res=TARGET_RESOLUTION):
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
    return cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


# ==========================================================
# Combined Pipeline (matches run_all_attacks in notebook)
# ==========================================================

def run_all_attacks(image):
    """Run all 3 noise variants + MSJPEG-Y, return dict of results."""
    maps = analyze_image(image)
    importance = maps["importance"]

    attacks = {}

    gaussian = apply_adaptive_gaussian(image, importance)
    attacks["Adaptive Gaussian"] = msjpeg_y_attack(gaussian)

    speckle = apply_adaptive_speckle(image, importance)
    attacks["Adaptive Speckle"] = msjpeg_y_attack(speckle)

    salt = apply_adaptive_salt_pepper(image, importance)
    attacks["Adaptive SaltPepper"] = msjpeg_y_attack(salt)

    return attacks


def combined_attack(image):
    """
    Run all 3 attacks, apply bilateral to each, return the one
    with highest PSNR (best visual quality since we can't check
    watermark without TrustMark decoder in production).
    """
    attacks = run_all_attacks(image)

    best_name = None
    best_psnr = -1
    best_img = None

    for name, atk_img in attacks.items():
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(
            atk_img, BILATERAL_DIAMETER,
            BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
        )

        # PSNR vs input as quality proxy
        psnr = cv2.PSNR(image, filtered)

        if psnr > best_psnr:
            best_psnr = psnr
            best_name = name
            best_img = filtered

    print(f"Selected: {best_name} (PSNR={best_psnr:.2f})")
    return best_img


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
