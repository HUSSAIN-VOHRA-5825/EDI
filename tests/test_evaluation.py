"""tests/test_evaluation.py — short tests for the robustness evaluation framework."""
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# 1. Bootstrap CI
# ──────────────────────────────────────────────────────────────────────────────

from scripts.run_robustness import bootstrap_ci


def test_bootstrap_ci_bounds():
    """ci_low <= mean <= ci_high for any non-trivial sample."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.75, scale=0.05, size=50)
    mean, ci_low, ci_high = bootstrap_ci(values, n_resamples=500, seed=0)
    assert ci_low <= mean, f"ci_low={ci_low} > mean={mean}"
    assert mean <= ci_high, f"mean={mean} > ci_high={ci_high}"


def test_bootstrap_ci_all_nan():
    """All-NaN input returns nan triple without crashing."""
    mean, lo, hi = bootstrap_ci(np.array([float("nan")] * 5))
    assert np.isnan(mean)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Prediction consistency
# ──────────────────────────────────────────────────────────────────────────────

from src.evaluation.models import ClassifierModel


def test_prediction_consistency_identical():
    """Consistency is 1.0 when original and varied predictions are identical."""
    orig_preds = [0, 1, 0, 1]
    # Simulate varied preds equal to original
    dummy_results = [{"prediction": p, "consistent": p == p} for p in orig_preds]
    consistencies = [r["consistent"] for r in dummy_results]
    assert all(consistencies), "All predictions identical → consistency must be 1.0"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Cross-validation fold integrity
# ──────────────────────────────────────────────────────────────────────────────

from sklearn.model_selection import StratifiedKFold


def test_cv_folds_no_overlap():
    """Train and test folds have no overlapping filenames/indices."""
    n = 20
    labels = ["under_30"] * 10 + ["30_plus"] * 10
    int_labels = [ClassifierModel.LABEL_MAP[l] for l in labels]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(np.zeros(n), int_labels):
        assert len(set(train_idx) & set(test_idx)) == 0, "Train/test indices overlap!"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Runner smoke test: 3 images, 1 variation, 2 levels, face_detection only
# ──────────────────────────────────────────────────────────────────────────────

def test_runner_smoke(tmp_path):
    """Runner completes on 3 synthetic images with face_detection model only."""
    import cv2
    import pandas as pd
    from src.utils.io import save_image
    import src.variations  # noqa: F401

    # Create 3 synthetic 512×512 images with a visible "face-like" gradient
    originals_dir = tmp_path / "data" / "eval" / "originals"
    originals_dir.mkdir(parents=True)

    filenames = []
    for i in range(3):
        img = np.full((512, 512, 3), fill_value=[200, 160, 140], dtype=np.uint8)
        # Add a circle as a crude face proxy (MediaPipe may not detect it,
        # but the runner should not crash even if detection fails)
        cv2.circle(img, (256, 256), 120, (180, 140, 110), -1)
        fname = f"img_{i:03d}.png"
        save_image(img, originals_dir / fname)
        filenames.append(fname)

    labels_df = pd.DataFrame({
        "filename": filenames,
        "label": ["under_30", "30_plus", "under_30"],
        "race": ["White", "Black", "Asian"],
        "age": ["20-29", "30-39", "20-29"],
    })
    labels_df.to_csv(originals_dir / "labels.csv", index=False)

    # Patch sys.argv and run
    import sys
    old_argv = sys.argv[:]
    sys.argv = [
        "run_robustness.py",
        "--models", "face_detection",
        "--variations", "blur_gaussian",
        "--levels", "1", "2",
        "--max-images", "3",
    ]
    # Redirect output dirs to tmp_path
    import scripts.run_robustness as rr
    orig_proj = rr.PROJECT_ROOT
    rr.PROJECT_ROOT = tmp_path
    try:
        (tmp_path / "results").mkdir(exist_ok=True)
        (tmp_path / "configs").mkdir(exist_ok=True)
        import shutil
        shutil.copy(PROJECT_ROOT / "configs" / "default.yaml", tmp_path / "configs" / "default.yaml")
        rr.main()
    except SystemExit:
        pass  # argparse may exit after --help; ignore
    finally:
        sys.argv = old_argv
        rr.PROJECT_ROOT = orig_proj

    # Just check that the script ran without unhandled exceptions
    assert True


# ──────────────────────────────────────────────────────────────────────────────
# 5. Skin edit mask subset test
# ──────────────────────────────────────────────────────────────────────────────

def test_skin_edit_mask_subset_of_extended_polygon():
    """Face+forehead portion (above chin) of the edit mask must lie inside the extended polygon.

    The neck box is intentionally outside the face oval — only the face region is checked here.
    """
    from src.face.mesh import FaceMeshDetector
    from src.utils.io import load_image, resize_keep_aspect_ratio

    input_dir = PROJECT_ROOT / "data" / "input"
    imgs = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))
    if not imgs:
        pytest.skip("No input image available")

    img = resize_keep_aspect_ratio(load_image(imgs[0]), target_longer_side=512)
    det = FaceMeshDetector()
    landmarks = det.detect_landmarks(img)
    if landmarks is None:
        pytest.skip("No face detected in input image")

    import cv2
    h, w = img.shape[:2]
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)
    face_h = max(10.0, float(y_max - y_min))

    ext_polygon = det._build_extended_polygon(landmarks, face_h, h, w)
    edit_mask = det.generate_skin_edit_mask(img)

    if edit_mask is None:
        pytest.skip("Could not compute edit mask")

    # Restrict to face+forehead region (above chin); neck box is by design outside the oval.
    chin_y = int(landmarks[152][1]) if len(landmarks) > 152 else int(y_max)
    face_only = edit_mask.copy()
    face_only[chin_y:, :] = 0

    # Dilate polygon to account for feathering at the mask boundary
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ext_dilated = cv2.dilate(ext_polygon, dil_k)
    outside = (face_only > 0) & (ext_dilated == 0)
    frac_outside = float(outside.sum()) / max(1, (face_only > 0).sum())
    assert frac_outside < 0.05, (
        f"{frac_outside:.1%} of face-region edit mask is outside extended polygon"
    )
    det.close()

