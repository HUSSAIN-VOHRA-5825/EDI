# Project: Controlled Face Image Variation Tool for Stress-Testing Medical AI

## Problem Statement
Medical AI systems must be robust to imaging conditions: lighting, blur, noise, JPEG compression, skin tone shifts, pose, and age. Collecting real images for every condition is impractical. This tool generates realistic, controlled variations of real face images across severity levels 1–5, checks that clinical identity is preserved, and stress-tests AI components via robustness curves (accuracy vs severity).

## Tech Stack
- Python 3.10+, OpenCV, NumPy, Pillow
- MediaPipe (Face Mesh landmarks + multiclass selfie segmenter)
- scikit-image (SSIM), PyTorch + lpips (perceptual similarity)
- facenet-pytorch (face embeddings / identity similarity)
- PyYAML (config), pandas (CSV), matplotlib (grids)
- pytest (tests); Streamlit (UI — future)

## Folder Layout
```
configs/default.yaml        # All parameters, thresholds, variation levels
data/input/                 # Reference face images
data/output/                # Generated images + metadata.csv  [gitignored]
data/eval/raw/              # FairFace 147-image evaluation subset  [gitignored]
examples/                   # One grid PNG per variation + metadata.csv (committed)
models/                     # Downloaded .tflite weight (auto-fetched by mesh.py) [gitignored]
src/core/                   # Base class, registry, config loader, pipeline runner
src/face/mesh.py            # Landmark detection, skin mask, edit mask
src/metrics/                # identity, ssim, lpips, detail_retention, ita, delta_e, preservation
src/variations/             # One file per variation family
src/utils/                  # IO, seeding, grid visualization
scripts/                    # CLI entry points (run_pipeline, make_grid, visualize_mask, make_eval_subset)
tests/                      # pytest suite
```

## Implemented Variations (11 real + 2 test dummies)
- **blur_gaussian** — Gaussian blur; sigma as fraction of image width
- **blur_motion** — Anti-aliased 4× supersampled line kernel; random angle; length as fraction of width
- **blur_defocus** — Circular disk (bokeh) kernel; radius as fraction of width
- **noise_gaussian** — Additive zero-mean Gaussian noise
- **noise_sensor** — Realistic CMOS sensor noise (Poisson shot + Gaussian read, in linear space)
- **jpeg_quality** — Lossy JPEG DCT quantization
- **jpeg_lowres** — Downscale + JPEG + bilinear upscale (low-res upload simulation)
- **light_lowlight** — Linear-space gain < 1; darkens with gamma correction
- **light_overexposed** — Linear-space gain > 1 with tanh shoulder rolloff
- **light_directional** — Smoothstep shadow gradient across the face
- **light_colortemp** — White-balance shift in linear sRGB (warm/cool)
- **skin_lighter / skin_darker** — Additive CIELAB L* shift with chroma compensation; feathered edit mask
- **skin_warm / skin_cool** — Additive CIELAB a*/b* shift; tuned to avoid purple (cool) and harsh cast (warm)

## Metrics and Gate Logic
Metrics computed per generated image (inside face mask unless noted): `identity_similarity` (FaceNet cosine), `ssim_face`, `ssim_full`, `lpips_face`, `detail_retention` (Laplacian variance ratio, clipped 0–2), `face_clipped_fraction`, `delta_e_face`, `ita_before/after`.  
Thresholds in `preservation_thresholds` in `default.yaml`. Gate passes when all active metrics meet thresholds. Skin-tone variations gate only on `identity`, `detail_retention`, and `clipped_fraction`.  
Level 1–2 pass for all variations; level 5 fails for `blur_motion`, `jpeg_lowres`, `noise_sensor`, `light_overexposed`.

## Coding Rules
- All parameters in YAML — no magic numbers in code.
- All randomness seeded from config (`seed: 42`).
- Effects applied to the whole image; the face mask is used only for metrics.
- Parameters resolution-invariant (fraction of image size, not fixed pixels).
- Never silently fail — log warnings for missing faces, failed models, etc.
- Clear docstrings and type hints; beginner-friendly comments explaining WHY.

## Status
**Done:** all 11 real variations + 2 test dummies, full preservation gate, face skin mask (MediaPipe multiclass segmenter + YCrCb fallback), FairFace 147-image evaluation subset.  
**Next:** robustness evaluation runner (`scripts/run_robustness.py`) — models under test (MediaPipe detection, FaceNet identity, ResNet18 classifier with 5-fold CV), robustness curves, per-race breakdown, `results/robustness_summary.md`.  
**After that:** pose/viewpoint variation, age simulation, Streamlit UI.
