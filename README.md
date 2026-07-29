# Watermark Removal — Classical Attack Pipeline

**Combined Pipeline:** Adaptive Gaussian Noise + Multi-Scale JPEG Y-channel Attack

Removes invisible watermarks (TrustMark/SynthID-style) using pure signal processing.  
No neural networks. No GPU. No ML models.

## Results

| Metric | Value |
|---|---|
| Confidence | -1.0 (fully removed) |
| SSIM | 0.9258 |
| PSNR | 33.56 dB |
| Deployment size | ~10 MB |

## How It Works

1. **Importance Map Analysis** — Texture + gradient + variance fusion identifies textured vs smooth regions
2. **Adaptive Gaussian Noise** — Adds perturbation guided by importance map (more in textured areas, less in smooth)
3. **MSJPEG-Y Attack** — Downscales Y channel to 128px, JPEG compresses at Q20, computes diff, upsamples back
4. **Bilateral Filter** — Edge-preserving cosmetic smoothing on final output

## Deploy

### Render
1. Push to GitHub
2. Go to render.com → New Web Service → Connect repo
3. It auto-detects `render.yaml` — just click Deploy

### Local
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## Tech Stack

- FastAPI + Uvicorn
- OpenCV (headless)
- NumPy + SciPy
- No PyTorch, no TensorFlow, no ONNX

## Authors

PES University, Bengaluru (RR Campus)
