# Project: Controlled Face Image Variation Tool for Stress-Testing Medical AI

## Problem Statement
Medical AI must be robust to imaging degradation (lighting, blur, noise, compression, skin tone shifts). Collecting real patient images for all conditions is impractical. This tool generates realistic, controlled variations of facial images across severity levels 1–5, evaluates clinical identity and skin-texture preservation, and benchmarks AI robustness without data collection.

## Tech Stack
- Python 3.10+, OpenCV, NumPy, Pillow, PyYAML, pandas, matplotlib, pytest
- MediaPipe (Face Mesh landmarks + multiclass selfie segmentation)
- PyTorch, torchvision, facenet-pytorch (InceptionResnetV1), lpips (AlexNet), scikit-image

## Folder Layout
- `configs/`: `default.yaml` (parameters, thresholds) and `chains.yaml` (chain definitions)
- `data/`: `input/` (reference image), `output/` (generated outputs), `eval/` (FairFace subset)
- `examples/`: Reference grids (levels 1–5) per variation/chain and `skin_edit_mask.png`
- `models/`: Auto-downloaded MediaPipe multiclass segmenter weights (`.tflite`)
- `src/`: `core/` (registry, pipeline), `face/` (mesh, skin masks), `metrics/`, `variations/`, `evaluation/`, `utils/`
- `scripts/`: CLI tools (`run_pipeline.py`, `run_robustness.py`, `make_grid.py`, `visualize_mask.py`, `plot_robustness.py`)
- `results/`: Robustness curves, heatmaps, summary reports, and examples

## Variations & Chains
- Blur: `blur_gaussian` (radius), `blur_motion` (linear smear), `blur_defocus` (aperture disk)
- Noise: `noise_gaussian` (additive electronic), `noise_sensor` (Poisson shot + Gaussian read)
- Compression: `jpeg_quality` (DCT quantization), `jpeg_lowres` (downsample + upscale)
- Lighting: `light_lowlight` (linear underexposure), `light_overexposed` (tanh roll-off), `light_directional` (shadow gradient), `light_colortemp` (warm/cool shift)
- Skin Tone: `skin_lighter`, `skin_darker`, `skin_warm`, `skin_cool` (CIE-LAB shifts on forehead/cheeks/nose)
- Chains: `chain_dim_clinic_phone`, `chain_shaky_handheld`, `chain_harsh_exam_lamp`, `chain_remote_upload`, `chain_skin_and_light`, `chain_random` (canonical order: skin -> light -> blur -> noise -> jpeg)

## Metrics & Gate Logic
- Metrics: Identity similarity (FaceNet cosine), SSIM (face/full), LPIPS, detail retention (Laplacian ratio), clipping fraction, delta-E, ITA.
- Gate thresholds: Identity >= 0.60, SSIM face >= 0.75, SSIM full >= 0.55, LPIPS face <= 0.08, detail >= 0.075.
- Evaluated against original image once per variation; failures are flagged in metadata notes but images are kept.
- Detail retention uses Laplacian variance ratio; ITA evaluates individual skin melanin tone.

## Coding Rules
- Deterministic execution via seeded NumPy `Generator`; never mutate inputs in-place.
- Strict canonical execution sequence in chains: `skin_tone -> lighting -> blur -> noise -> jpeg`.
- Preservation checks operate on facial metric mask; variations never touch metric mask logic.

## Status & Known Limitations
- Status: 15 atomic variations, 6 chains, and Streamlit demo app completed; quick robustness evaluation implemented. Next: full evaluation, age variation.
- Known limitations: detail_retention is inflated by noise; ITA is only meaningful for skin tone variations.
