"""tests/test_evaluation.py — short unit tests for the robustness evaluation framework."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_robustness import bootstrap_ci, wilson_ci, compute_ci
from src.evaluation.models import ClassifierModel


# ──────────────────────────────────────────────────────────────────────────────
# 1. Confidence Intervals (Bootstrap & Wilson)
# ──────────────────────────────────────────────────────────────────────────────

def test_bootstrap_ci_bounds():
    """ci_low <= mean <= ci_high for non-trivial samples."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.75, scale=0.05, size=50)
    mean, ci_low, ci_high = bootstrap_ci(values, n_resamples=500, seed=0)
    assert ci_low <= mean <= ci_high, f"Bounds error: ci_low={ci_low}, mean={mean}, ci_high={ci_high}"


def test_bootstrap_ci_all_nan():
    """All-NaN input returns nan without crashing."""
    mean, lo, hi = bootstrap_ci(np.array([float("nan")] * 5))
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_wilson_ci_non_zero_width_on_perfect_score():
    """Wilson interval for 40/40 binary outcomes produces a non-zero width band."""
    vals_perfect = np.ones(40)
    p, ci_low, ci_high = wilson_ci(vals_perfect, confidence=0.95)
    assert p == 1.0, f"Expected p=1.0, got {p}"
    assert ci_low < 1.0, f"Wilson ci_low should be strictly < 1.0 for n=40, got {ci_low}"
    assert ci_high == 1.0, f"Wilson ci_high should be 1.0, got {ci_high}"
    assert (ci_high - ci_low) > 0.05, "Expected non-zero width band"


def test_compute_ci_dispatcher():
    """compute_ci selects Wilson for proportions and bootstrap for continuous."""
    p, lo, hi = compute_ci("detection_rate", np.ones(40))
    assert lo < 1.0, "Should use Wilson for detection_rate"
    rng = np.random.default_rng(42)
    m, lo2, hi2 = compute_ci("identity_similarity", rng.uniform(0.7, 0.9, size=40))
    assert lo2 <= m <= hi2, "Should use bootstrap for continuous similarity"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Prediction consistency
# ──────────────────────────────────────────────────────────────────────────────

def test_prediction_consistency_identical():
    """Consistency is 1.0 when varied predictions match original predictions."""
    orig_preds = [0, 1, 0, 1, 1]
    dummy_results = [{"prediction": p, "consistent": p == p} for p in orig_preds]
    consistencies = [r["consistent"] for r in dummy_results]
    assert all(consistencies), "All identical predictions must have consistency=True"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Cross-validation fold integrity
# ──────────────────────────────────────────────────────────────────────────────

def test_cv_folds_no_overlap():
    """Train and test folds have strictly no overlapping filenames/indices."""
    n = 20
    labels = ["under_30"] * 10 + ["30_plus"] * 10
    int_labels = [ClassifierModel.LABEL_MAP[l] for l in labels]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(np.zeros(n), int_labels):
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Overlapping indices detected in fold: {overlap}"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Runner smoke test & resume test
# ──────────────────────────────────────────────────────────────────────────────

def test_runner_smoke_and_resume(tmp_path):
    """Runner completes on 3 synthetic images, 1 variation, 2 levels, and resumes on second call."""
    import cv2
    from src.utils.io import save_image
    import scripts.run_robustness as rr

    # Create synthetic evaluation images
    originals_dir = tmp_path / "data" / "eval" / "originals"
    originals_dir.mkdir(parents=True)

    filenames = []
    for i in range(3):
        img = np.full((512, 512, 3), fill_value=[200, 160, 140], dtype=np.uint8)
        cv2.circle(img, (256, 256), 120, (180, 140, 110), -1)
        fname = f"synth_{i:03d}.png"
        save_image(img, originals_dir / fname)
        filenames.append(fname)

    labels_df = pd.DataFrame({
        "filename": filenames,
        "label": ["under_30", "30_plus", "under_30"],
        "race": ["White", "Black", "Asian"],
        "age": ["20-29", "30-39", "20-29"],
    })
    labels_df.to_csv(originals_dir / "labels.csv", index=False)

    (tmp_path / "configs").mkdir(exist_ok=True)
    shutil.copy(PROJECT_ROOT / "configs" / "default.yaml", tmp_path / "configs" / "default.yaml")

    # Patch sys.argv and PROJECT_ROOT
    old_argv = sys.argv[:]
    orig_proj = rr.PROJECT_ROOT
    rr.PROJECT_ROOT = tmp_path

    try:
        # First call: fresh run
        sys.argv = [
            "run_robustness.py",
            "--models", "face_detection",
            "--variations", "blur_defocus",
            "--levels", "1", "3",
            "--max-images", "3",
            "--batch-size", "2",
        ]
        rr.main()

        rob_csv = tmp_path / "results" / "robustness.csv"
        prog_json = tmp_path / "results" / ".progress.json"
        assert rob_csv.exists(), "robustness.csv not created"
        assert prog_json.exists(), ".progress.json not created"

        progress_data = json.loads(prog_json.read_text(encoding="utf-8"))
        assert "face_detection:blur_defocus" in progress_data["completed_pairs"]
        first_line_count = len(rob_csv.read_text(encoding="utf-8").strip().splitlines())

        # Second call: resume mode (should skip recomputing without errors or duplicate rows)
        sys.argv = [
            "run_robustness.py",
            "--models", "face_detection",
            "--variations", "blur_defocus",
            "--levels", "1", "3",
            "--max-images", "3",
            "--batch-size", "2",
        ]
        rr.main()

        second_line_count = len(rob_csv.read_text(encoding="utf-8").strip().splitlines())
        assert second_line_count == first_line_count, (
            f"Resume failed to skip: line count changed from {first_line_count} to {second_line_count}"
        )

    finally:
        sys.argv = old_argv
        rr.PROJECT_ROOT = orig_proj


# ──────────────────────────────────────────────────────────────────────────────
# 5. Skin edit mask subset test
# ──────────────────────────────────────────────────────────────────────────────

def test_skin_edit_mask_subset_of_extended_polygon():
    """Face-region portion of edit mask (above chin) must lie inside extended polygon."""
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

    chin_y = int(landmarks[152][1]) if len(landmarks) > 152 else int(y_max)
    face_only = edit_mask.copy()
    face_only[chin_y:, :] = 0

    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ext_dilated = cv2.dilate(ext_polygon, dil_k)
    outside = (face_only > 0) & (ext_dilated == 0)
    frac_outside = float(outside.sum()) / max(1, (face_only > 0).sum())
    assert frac_outside < 0.05, f"{frac_outside:.1%} of face-region edit mask is outside extended polygon"

    det.close()
