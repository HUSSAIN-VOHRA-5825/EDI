"""app/helpers.py

Helper utilities for the Streamlit demo application:
- Formatting preservation metrics into clean tabular displays
- Plain-language explanations for failed preservation checks
- Safe no-face detection messaging
- In-memory ZIP package creation for batch generation
- Face skin mask overlay rendering
"""

from datetime import datetime
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import zipfile
import cv2
import numpy as np
import pandas as pd


METRIC_EXPLANATIONS = {
    "identity": "FaceNet embedding cosine similarity fell below 0.60 (identity not preserved).",
    "ssim_face": "Structural similarity in the facial region dropped below the 0.75 threshold.",
    "ssim_full": "Whole-image structural similarity fell below the 0.55 threshold.",
    "lpips_face": "Perceptual difference in facial features exceeded the 0.08 threshold.",
    "detail_retention": "High-frequency dermatological texture (Laplacian variance) dropped below 0.075.",
    "face_clipped_fraction": "More than 5% of facial skin pixels were clipped (0 or 255) by saturation/crushing.",
}


def get_no_face_message() -> str:
    """Return friendly message when MediaPipe cannot find a face in the input image."""
    return "No face detected in the image. Please upload a clear, front-facing portrait."


def get_failed_check_explanations(passes: Dict[str, bool]) -> List[str]:
    """Return plain-language explanations for all active checks that failed.

    Args:
        passes: Dictionary mapping check name to bool pass/fail status.

    Returns:
        List of one-line plain language strings explaining each failure.
    """
    failures = []
    for check_name, passed in passes.items():
        if not passed:
            explanation = METRIC_EXPLANATIONS.get(
                check_name,
                f"Metric '{check_name}' failed the clinical preservation threshold."
            )
            failures.append(explanation)
    return failures


def format_metrics_table(
    metrics: Dict[str, Any],
    variation_name: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Format quantitative preservation metrics into a presentation-ready DataFrame.

    ITA (Individual Typology Angle) is included ONLY for skin tone variations (skin_*).

    Args:
        metrics: Dictionary returned by check_preservation.
        variation_name: Name of the executed variation.
        thresholds: Optional thresholds dictionary.

    Returns:
        pd.DataFrame with columns: ['Metric', 'Value', 'Threshold', 'Status']
    """
    th = thresholds or {}
    id_min = th.get("identity_similarity_min", 0.60)
    ssim_face_min = th.get("ssim_face_min", 0.75)
    ssim_full_min = th.get("ssim_full_min", 0.55)
    lpips_max = th.get("lpips_face_max", 0.08)
    detail_min = th.get("min_detail_retention", 0.075)
    clip_max = th.get("max_face_clipped_fraction", 0.05)

    passes = metrics.get("passes", {})

    rows = [
        {
            "Metric": "Identity Similarity (FaceNet cosine)",
            "Value": f"{metrics.get('identity_similarity', 0.0):.4f}",
            "Threshold": f"≥ {id_min:.2f}",
            "Status": "Within" if passes.get("identity", True) else "Beyond",
        },
        {
            "Metric": "SSIM (Face Region)",
            "Value": f"{metrics.get('ssim_face', 0.0):.4f}",
            "Threshold": f"≥ {ssim_face_min:.2f}",
            "Status": "Within" if passes.get("ssim_face", True) else "Beyond",
        },
        {
            "Metric": "SSIM (Full Image)",
            "Value": f"{metrics.get('ssim_full', 0.0):.4f}",
            "Threshold": f"≥ {ssim_full_min:.2f}",
            "Status": "Within" if passes.get("ssim_full", True) else "Beyond",
        },
        {
            "Metric": "LPIPS Distance (Face Region)",
            "Value": f"{metrics.get('lpips_face', 0.0):.4f}",
            "Threshold": f"≤ {lpips_max:.2f}",
            "Status": "Within" if passes.get("lpips_face", True) else "Beyond",
        },
        {
            "Metric": "Detail Retention (Laplacian Ratio)",
            "Value": f"{metrics.get('detail_retention', 0.0):.4f}",
            "Threshold": f"≥ {detail_min:.3f}",
            "Status": "Within" if passes.get("detail_retention", True) else "Beyond",
        },
        {
            "Metric": "Delta E (Face Region)",
            "Value": f"{metrics.get('delta_e_face', 0.0):.2f}",
            "Threshold": "Information only",
            "Status": "INFO",
        },
        {
            "Metric": "Face Clipped Fraction",
            "Value": f"{metrics.get('face_clipped_fraction', 0.0):.2%}",
            "Threshold": f"≤ {clip_max:.1%}",
            "Status": "Within" if passes.get("face_clipped_fraction", True) else "Beyond",
        },
    ]

    # Include ITA ONLY for skin tone variations (not meaningful for other degradations)
    if variation_name.startswith("skin_"):
        ita_before = metrics.get("ita_before")
        ita_after = metrics.get("ita_after")
        ita_before_str = f"{ita_before:.1f}°" if ita_before is not None else "N/A"
        ita_after_str = f"{ita_after:.1f}°" if ita_after is not None else "N/A"
        rows.append({
            "Metric": "ITA (Individual Typology Angle)",
            "Value": f"{ita_before_str} → {ita_after_str}",
            "Threshold": "Clinical monitoring",
            "Status": "INFO",
        })

    return pd.DataFrame(rows)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 230, 115),
    alpha: float = 0.35,
) -> np.ndarray:
    """Create a semi-transparent color overlay of the face mask on the RGB image.

    Args:
        image: RGB uint8 image (H, W, 3).
        mask: Binary mask uint8 (H, W).
        color: RGB tuple for overlay color.
        alpha: Overlay opacity (0.0 to 1.0).

    Returns:
        np.ndarray: Blended RGB image.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return image.copy()

    overlay = image.copy()
    face_px = mask > 0
    colored_layer = np.zeros_like(image)
    colored_layer[:] = color

    blended = (image.astype(np.float32) * (1.0 - alpha) + colored_layer.astype(np.float32) * alpha).astype(np.uint8)
    overlay[face_px] = blended[face_px]

    # Draw a thin boundary contour for visual crispness
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 1)

    return overlay


def create_batch_zip(
    records: List[Dict[str, Any]],
    images_dict: Dict[str, np.ndarray],
) -> bytes:
    """Pack batch result images and metadata.csv into an in-memory ZIP archive.

    Args:
        records: List of metadata dictionaries.
        images_dict: Mapping of relative file path strings to uint8 RGB images.

    Returns:
        bytes: Compressed ZIP file bytes ready for download.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add metadata CSV
        if records:
            df = pd.DataFrame(records)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            zf.writestr("metadata.csv", csv_bytes)

        # 2. Add generated images
        for path_name, img_arr in images_dict.items():
            # Convert RGB array to PNG bytes
            is_success, encoded_img = cv2.imencode(".png", cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR))
            if is_success:
                zf.writestr(path_name, encoded_img.tobytes())

    return buf.getvalue()


def get_run_info_caption(results_dir: Union[Path, str]) -> str:
    """Generate summary caption from results/run_info.json and results/robustness.csv.

    Args:
        results_dir: Path to directory containing run_info.json and robustness.csv.

    Returns:
        Formatted caption string or 'no run information found'.
    """
    res_path = Path(results_dir)
    run_info_file = res_path / "run_info.json"
    if not run_info_file.is_file():
        return "no run information found"

    try:
        with open(run_info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception:
        return "no run information found"

    n_images = info.get("images_evaluated", "unknown")
    levels = info.get("levels_evaluated", [])
    levels_str = ", ".join(str(lvl) for lvl in levels) if levels else "none"

    mtime = run_info_file.stat().st_mtime
    date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    csv_file = res_path / "robustness.csv"
    has_classifier = False
    if csv_file.is_file():
        try:
            df = pd.read_csv(csv_file)
            if "model" in df.columns and "classifier" in df["model"].values:
                has_classifier = True
        except Exception:
            pass

    if not has_classifier:
        classifier_part = ", classifier not yet evaluated"
    else:
        classifier_part = ""

    return f"Evaluated n={n_images} images per model across levels [{levels_str}], generated {date_str}{classifier_part}."
