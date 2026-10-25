"""Preservation checking harness for clinical and identity metrics.

Integrates:
  - Identity preservation (FaceNet cosine similarity)
  - Full structural similarity (SSIM, entire image)
  - Facial skin structural similarity (SSIM on skin mask)
  - Perceptual distortion (LPIPS on skin mask)
  - Face-region pixel clipping fraction (crushed to 0 or saturated to 255)
  - Detail retention (Laplacian variance ratio inside face mask)  [GATE]
  - Mean CIE 1976 ΔE inside face mask                             [INFO ONLY]

Pass/fail gate: identity, ssim_face, ssim_full, lpips_face, detail_retention.
delta_e_face is recorded and returned but never blocks a pass.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from src.metrics.identity import identity_similarity
from src.metrics.lpips_metric import lpips_score
from src.metrics.ssim import ssim_score
from src.metrics.detail_retention import detail_retention as compute_detail_retention
from src.metrics.delta_e import delta_e_face as compute_delta_e


# Default fallback thresholds (calibrated across all 11 variations)
DEFAULT_THRESHOLDS = {
    "identity_similarity_min": 0.60,   # FaceNet cosine sim floor
    "ssim_face_min": 0.75,             # Face-region structural similarity
    "ssim_full_min": 0.55,             # Whole-image structural similarity
    "lpips_face_max": 0.08,            # Face-region perceptual distance
    "min_detail_retention": 0.075,     # Fine skin texture Laplacian ratio
}


def compute_face_clipped_fraction(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Calculate the fraction of pixels inside the face mask where any RGB channel is 0 or 255.

    In medical imaging and clinical facial AI, sensor clipping (over-exposure / blowout at 255
    or under-exposure / underexposed crushing at 0) destroys dermatological texture,
    micro-vascular features, and lesion borders.

    Args:
        image: RGB image as uint8 NumPy array (H, W, 3).
        mask: Optional binary skin mask (H, W). If provided, calculates clipping exclusively
            inside positive mask pixels (mask > 0). If None, calculates over the entire image.

    Returns:
        float: Fraction between 0.0 and 1.0 representing clipped pixels.
    """
    if mask is not None and np.count_nonzero(mask > 0) > 0:
        face_pixels = image[mask > 0]
    else:
        face_pixels = image.reshape(-1, image.shape[-1])

    # A pixel is clipped if ANY channel is saturated at 255 or crushed at 0
    is_clipped = np.any((face_pixels == 0) | (face_pixels == 255), axis=-1)
    return float(np.mean(is_clipped))


def check_preservation(
    original: np.ndarray,
    generated: np.ndarray,
    mask: Optional[np.ndarray] = None,
    thresholds: Optional[Dict[str, float]] = None,
    gate_metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate all preservation metrics and verify adherence to thresholds.

    Args:
        original: Reference RGB image array (H, W, 3).
        generated: Modified/varied RGB image array (H, W, 3).
        mask: Optional binary skin mask (H, W) for region-restricted metrics.
        thresholds: Dictionary of acceptance thresholds. If None, uses DEFAULT_THRESHOLDS.
        gate_metrics: Optional list of metric keys to enforce in the pass/fail gate.
            If None, enforces standard default gate ['identity', 'ssim_face', 'ssim_full', 'lpips_face', 'detail_retention'].

    Returns:
        Dict containing:
            - 'identity_similarity': float
            - 'ssim_full': float
            - 'ssim_face': float
            - 'lpips_face': float
            - 'face_clipped_fraction': float
            - 'detail_retention': float
            - 'delta_e_face': float
            - 'ita_before': float
            - 'ita_after': float
            - 'passes': Dict[str, bool] containing pass/fail flags for active gate metrics
            - 'passed': bool (True only if ALL active gate checks pass)
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    from src.metrics.ita import compute_ita

    # --- 1. Compute quantitative metrics ---
    id_sim = identity_similarity(original, generated)
    ssim_full_val = ssim_score(original, generated, mask=None)
    ssim_face_val = ssim_score(original, generated, mask=mask)
    lpips_face_val = lpips_score(original, generated, mask=mask)
    face_clipped_frac = compute_face_clipped_fraction(generated, mask=mask)
    detail_ret = compute_detail_retention(original, generated, mask=mask)
    de_face = compute_delta_e(original, generated, mask=mask)
    ita_orig = compute_ita(original, mask=mask)
    ita_gen = compute_ita(generated, mask=mask)

    # --- 2. Evaluate gate metrics against thresholds ---
    id_min = thresholds.get("identity_similarity_min", DEFAULT_THRESHOLDS["identity_similarity_min"])
    ssim_face_min = thresholds.get("ssim_face_min", DEFAULT_THRESHOLDS["ssim_face_min"])
    ssim_full_min = thresholds.get("ssim_full_min", DEFAULT_THRESHOLDS["ssim_full_min"])
    lpips_max = thresholds.get("lpips_face_max", DEFAULT_THRESHOLDS["lpips_face_max"])
    detail_min = thresholds.get("min_detail_retention", DEFAULT_THRESHOLDS["min_detail_retention"])
    max_clip = thresholds.get("max_face_clipped_fraction", 0.05)

    all_passes = {
        "identity": bool(id_sim >= id_min),
        "ssim_face": bool(ssim_face_val >= ssim_face_min),
        "ssim_full": bool(ssim_full_val >= ssim_full_min),
        "lpips_face": bool(lpips_face_val <= lpips_max),
        "detail_retention": bool(detail_ret >= detail_min),
        "clipped_fraction": bool(face_clipped_frac <= max_clip),
    }

    if gate_metrics is not None:
        active_passes = {k: v for k, v in all_passes.items() if k in gate_metrics}
    else:
        default_keys = ["identity", "ssim_face", "ssim_full", "lpips_face", "detail_retention"]
        active_passes = {k: v for k, v in all_passes.items() if k in default_keys}

    overall_passed = bool(all(active_passes.values()))

    return {
        "identity_similarity": round(id_sim, 4),
        "ssim_full": round(ssim_full_val, 4),
        "ssim_face": round(ssim_face_val, 4),
        "lpips_face": round(lpips_face_val, 4),
        "face_clipped_fraction": round(face_clipped_frac, 4),
        "detail_retention": round(detail_ret, 4),
        "delta_e_face": de_face,
        "ita_before": ita_orig,
        "ita_after": ita_gen,
        "passes": active_passes,
        "passed": overall_passed,
    }
