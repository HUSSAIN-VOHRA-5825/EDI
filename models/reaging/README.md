# FRAN Face Re-aging Model Assets

This directory contains assets for the FRAN (Face Re-Aging Network) ONNX model used in the Face Variation Pipeline.

## Files in this Directory

- `mask512.jpg` (tracked in Git): High-resolution boundary blend mask for 512×512 facial patches.
- `mask1024.jpg` (tracked in Git): Boundary blend mask for 1024×1024 patches.
- `unet.onnx` (auto-generated / downloaded): Exported ONNX U-Net model (~124.2 MB) used for CPU inference.

> [!NOTE]
> `unet.onnx` is 124.2 MB, which exceeds GitHub's 100 MB single-file limit for regular git files. To run locally or set up from fresh clones, the model is automatically downloaded/exported using the script below.

## Automated Download & Setup

To fetch the pretrained PyTorch weights and export `unet.onnx` locally with a single command:

```bash
python scripts/download_reaging_model.py
```

The pipeline (`src/face/aging.py`) also calls this automatically on first use if `unet.onnx` is missing.
