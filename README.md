# Controlled Face Image Variation Tool for Stress-Testing Medical AI

A modular framework that generates physically motivated, controlled variations of facial images across severity levels 1–5, while quantifying identity and clinical skin-region preservation. Used to stress-test medical AI systems without collecting large amounts of real data.

---

## Folder Layout

```
project_root/
├── configs/default.yaml         # All parameters, thresholds, and variation levels
├── data/                        # Input images, evaluation subset, and generated variations
│   ├── input/                   # Reference face images (place your images here)
│   ├── output/                  # Generated images, grids, metadata.csv
│   └── eval/raw/                # FairFace evaluation subset — 147 images + labels.csv
├── examples/                    # Generated example grids
├── models/                      # MediaPipe selfie segmenter model weights
│   └── selfie_multiclass_256x256.tflite  # MediaPipe multiclass selfie segmenter
│                                # Downloaded automatically from:
│                                # https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
├── src/
│   ├── core/                    # Base class, registry, config loader, pipeline runner
│   ├── face/                    # MediaPipe Face Mesh + skin mask extraction (mesh.py)
│   ├── metrics/                 # Identity, SSIM, LPIPS, detail retention, ITA, delta-E
│   ├── variations/              # One file per variation family
│   └── utils/                   # Image IO, seeding, matplotlib grid visualization
├── scripts/
│   ├── run_pipeline.py          # Run all variations on data/input/; writes data/output/
│   ├── make_grid.py             # Generate before/after grid for one image + variation
│   ├── visualize_mask.py        # Overlay facial skin mask on an image
│   └── make_eval_subset.py      # Build FairFace evaluation subset (run once)
└── tests/                       # pytest test suite
```

This repository includes the current input image, evaluation data, generated variations, example grids, and model weights. Review the image dataset's license and usage terms before redistributing or reusing those assets. Add additional face images under `data/input/`.

---

## Environment Setup

```bash
conda activate edi_env          # or use: C:\Users\vohra\anaconda3\envs\edi_env\python.exe
pip install -r requirements.txt
```

*Note on MediaPipe:* use `mediapipe>=0.10.14,<0.11` for `mp.solutions.face_mesh` compatibility.

The `models/selfie_multiclass_256x256.tflite` file is downloaded automatically on first use of the skin mask (see `src/face/mesh.py → _ensure_segmenter_model`).

---

## Quickstart

```bash
# Run all registered variations on data/input/ images
python scripts/run_pipeline.py

# Generate a grid for one image and one variation
python scripts/make_grid.py --image data/input/person_1.jpeg --variation blur_gaussian

# Inspect skin mask
python scripts/visualize_mask.py

# Run tests
pytest tests/ -v
```

---

## Adding Images

Place `.png`, `.jpg`, or `.jpeg` face images in `data/input/`. The pipeline resizes to 512 px (longer side, Lanczos) and processes every image found.
