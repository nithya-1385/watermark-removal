# Watermark Removal — Multi-Resolution MSJPEG-Y Attack

**Pipeline:** Multi-Resolution JPEG Y-channel Attack (no neural networks, no GPU)

Removes invisible watermarks (TrustMark/SynthID-style) by averaging JPEG compression
diffs across 5 internal resolutions. Block edges from each resolution cancel out,
eliminating visible grid artifacts while maintaining strong watermark removal.

## Results

| Metric | Value |
|---|---|
| Removal rate (1000 diverse images) | **83%** |
| Removal rate (1000 embeddings, same image) | **100%** |
| Avg quality retention (SSIM) | **95.96%** |
| Avg PSNR | **33+ dB** |
| Deployment size | **~10 MB** |

## How It Works

1. **Multi-Resolution Y-channel JPEG** — Downscales the Y (luminance) channel to 5
   different internal resolutions [96, 112, 128, 144, 160px], applies JPEG compression
   at Q=20 to each, computes the diff, upsamples back to full resolution, and averages
   all 5 diffs together. Block edges from each resolution land at different pixel
   positions and cancel out — no visible grid artifacts.
2. **Bilateral Filter** — Edge-preserving cosmetic smoothing on final output.
3. **Alpha channel support** — PNG transparency is preserved throughout.

## Why This Works

TrustMark (used as a proxy for SynthID) embeds watermarks in the frequency domain.
JPEG compression at low quality disrupts these frequency components. By averaging across
multiple internal resolutions, the attack is both stronger (wider frequency disruption)
and cleaner (no single resolution's block pattern dominates).

## Validated On

- **1000 COCO val2017 images** — diverse content: people, animals, food, indoor/outdoor
- **15 curated test images** — various textures, resolutions, and content types
- **TrustMark-Q** as SynthID proxy (confirmed same architecture per Gowal et al., 2025)

## Deploy

### Render (one click)
1. Push to GitHub
2. Go to render.com → New Web Service → Connect repo
3. Auto-detects `render.yaml` → click Deploy

### Local
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## Tech Stack

- FastAPI + Uvicorn
- OpenCV (headless)
- NumPy
- No PyTorch, no TensorFlow, no ML models in production

## References

| Paper | Relevance |
|---|---|
| Gowal et al., arXiv:2510.09263, 2025 | SynthID-Image — confirms post-hoc encoder-decoder |
| Bui et al., ICCV 2023 | TrustMark — proxy watermarking system |
| Kassis & Hengartner, IEEE S&P 2025 | UnMarker — theoretical basis for classical attacks |

Proof of Concept Research project
