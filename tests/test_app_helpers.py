"""tests/test_app_helpers.py

Unit tests for app/helpers.py:
- format_metrics_table converts check_preservation dictionary into structured DataFrame
- get_failed_check_explanations returns plain-language explanations for failures
- get_no_face_message returns friendly notification when face detection fails
- ITA metric is included for skin_* variations and strictly omitted for non-skin variations
- create_batch_zip generates valid ZIP archive containing metadata.csv and PNGs
"""

import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import pytest

from app.helpers import (
    create_batch_zip,
    format_metrics_table,
    get_failed_check_explanations,
    get_no_face_message,
    get_run_info_caption,
    overlay_mask_on_image,
)


@pytest.fixture
def sample_metrics() -> dict:
    """Mock metrics dictionary returned by check_preservation."""
    return {
        "identity_similarity": 0.825,
        "ssim_face": 0.780,
        "ssim_full": 0.650,
        "lpips_face": 0.055,
        "face_clipped_fraction": 0.012,
        "detail_retention": 0.095,
        "delta_e_face": 3.45,
        "ita_before": 42.0,
        "ita_after": 28.5,
        "passes": {
            "identity": True,
            "ssim_face": True,
            "ssim_full": True,
            "lpips_face": True,
            "detail_retention": True,
            "face_clipped_fraction": True,
        },
        "passed": True,
    }


@pytest.fixture
def failing_metrics() -> dict:
    """Mock metrics dictionary with multiple threshold violations."""
    return {
        "identity_similarity": 0.450,
        "ssim_face": 0.620,
        "ssim_full": 0.400,
        "lpips_face": 0.120,
        "face_clipped_fraction": 0.080,
        "detail_retention": 0.040,
        "delta_e_face": 12.5,
        "ita_before": None,
        "ita_after": None,
        "passes": {
            "identity": False,
            "ssim_face": False,
            "ssim_full": False,
            "lpips_face": False,
            "detail_retention": False,
            "face_clipped_fraction": False,
        },
        "passed": False,
    }


def test_format_metrics_table_structure(sample_metrics: dict) -> None:
    """Verify that format_metrics_table returns DataFrame with required columns and values."""
    df = format_metrics_table(sample_metrics, variation_name="blur_motion")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Metric", "Value", "Threshold", "Status"]
    assert len(df) == 7  # 7 core metrics for non-skin variations

    metric_names = df["Metric"].tolist()
    assert any("Identity Similarity" in m for m in metric_names)
    assert any("SSIM (Face Region)" in m for m in metric_names)
    assert any("LPIPS" in m for m in metric_names)
    assert any("Detail Retention" in m for m in metric_names)
    assert any("Delta E" in m for m in metric_names)
    assert any("Face Clipped Fraction" in m for m in metric_names)

    # Core non-info metrics should show "Within" for passing checks
    non_info = df[df["Status"] != "INFO"]
    assert (non_info["Status"] == "Within").all()


def test_format_metrics_table_failing_labels(failing_metrics: dict) -> None:
    """Verify that failing gate checks receive 'Beyond' status label."""
    df = format_metrics_table(failing_metrics, variation_name="blur_motion")
    non_info = df[df["Status"] != "INFO"]
    assert (non_info["Status"] == "Beyond").all()


def test_ita_omitted_for_non_skin_variations(sample_metrics: dict) -> None:
    """ITA metric must be omitted for non-skin variations (e.g. blur, noise, jpeg, light, chains)."""
    non_skin_names = [
        "blur_motion",
        "blur_gaussian",
        "noise_sensor",
        "jpeg_lowres",
        "light_lowlight",
        "chain_dim_clinic_phone",
    ]

    for name in non_skin_names:
        df = format_metrics_table(sample_metrics, variation_name=name)
        ita_rows = df[df["Metric"].str.contains("ITA", case=False)]
        assert len(ita_rows) == 0, f"ITA unexpectedly present for non-skin variation {name}"


def test_ita_included_for_skin_variations(sample_metrics: dict) -> None:
    """ITA metric must be included for skin_* variations."""
    skin_names = ["skin_darker", "skin_lighter", "skin_warm", "skin_cool"]

    for name in skin_names:
        df = format_metrics_table(sample_metrics, variation_name=name)
        ita_rows = df[df["Metric"].str.contains("ITA", case=False)]
        assert len(ita_rows) == 1, f"ITA missing for skin variation {name}"
        val = ita_rows.iloc[0]["Value"]
        assert "42.0°" in val and "28.5°" in val


def test_get_failed_check_explanations(failing_metrics: dict) -> None:
    """Verify plain-language explanations are produced for each failed gate metric."""
    explanations = get_failed_check_explanations(failing_metrics["passes"])

    assert len(explanations) == 6
    full_text = " ".join(explanations)
    assert "FaceNet embedding" in full_text
    assert "Structural similarity" in full_text
    assert "Perceptual difference" in full_text
    assert "texture" in full_text or "Laplacian" in full_text
    assert "clipped" in full_text or "saturation" in full_text


def test_get_no_face_message() -> None:
    """Verify safe, non-crashing no-face message."""
    msg = get_no_face_message()
    assert isinstance(msg, str)
    assert "No face detected" in msg


def test_overlay_mask_on_image() -> None:
    """Verify mask overlay function preserves dimensions and returns uint8."""
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255

    blended = overlay_mask_on_image(img, mask)
    assert blended.shape == (64, 64, 3)
    assert blended.dtype == np.uint8
    # Pixels inside mask should have changed
    assert not np.array_equal(blended[30, 30], img[30, 30])
    # Pixels outside mask (away from border) should be unchanged
    assert np.array_equal(blended[5, 5], img[5, 5])


def test_create_batch_zip() -> None:
    """Verify in-memory ZIP package creation with metadata.csv and PNG images."""
    records = [
        {"original_filename": "test.jpg", "variation": "blur_motion", "severity": 3, "passed": True}
    ]
    images = {
        "blur_motion/level_3/test.png": np.zeros((32, 32, 3), dtype=np.uint8)
    }

    zip_bytes = create_batch_zip(records, images)
    assert len(zip_bytes) > 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "metadata.csv" in namelist
        assert "blur_motion/level_3/test.png" in namelist


def test_get_run_info_caption(tmp_path: Path) -> None:
    """Test get_run_info_caption for missing files, and runs with/without classifier."""
    # 1. Missing run_info.json
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    assert get_run_info_caption(empty_dir) == "no run information found"

    # 2. Existing run_info.json without classifier in robustness.csv
    run_dir = tmp_path / "run_results"
    run_dir.mkdir()
    info_path = run_dir / "run_info.json"
    info_path.write_text(
        json.dumps({
            "images_evaluated": 40,
            "levels_evaluated": [1, 3, 5],
            "models_evaluated": ["face_detection", "identity_model"],
        }),
        encoding="utf-8",
    )
    caption_no_clf = get_run_info_caption(run_dir)
    assert "n=40" in caption_no_clf
    assert "[1, 3, 5]" in caption_no_clf
    assert "classifier not yet evaluated" in caption_no_clf

    # 3. Existing run_info.json with classifier rows in robustness.csv
    csv_path = run_dir / "robustness.csv"
    csv_path.write_text(
        "model,variation,level,metric,value,n\nclassifier,blur_motion,1,accuracy,0.85,40\n",
        encoding="utf-8",
    )
    caption_with_clf = get_run_info_caption(run_dir)
    assert "classifier not yet evaluated" not in caption_with_clf
    assert "n=40" in caption_with_clf
