"""Detail-retention metric based on Laplacian variance.

Quantifies how much fine-grained texture (skin pores, hair follicles,
lesion edges) the variation preserves inside the face mask.

SSIM and LPIPS are intensity-based and can miss the case where a
moderate SSIM image has had all high-frequency skin texture smoothed
away (as seen in jpeg_lowres level 5).

Metric definition
-----------------
  detail_retention = clamp(var_lap(generated) / var_lap(original), 0, 2)

  where var_lap(x) = variance of the Laplacian of the grayscale image,
  computed ONLY over pixels inside the face mask.

  * Value 1.0  → texture exactly preserved.
  * Value < 1  → texture smoothed / blurred away.
  * Value > 1  → extra high-frequency noise (sharpening or sensor noise).
  * Clamped to [0, 2] to avoid unbounded outputs.
"""

from typing import Optional
import cv2
import numpy as np


def detail_retention(
    original: np.ndarray,
    generated: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> float:
    """Compute the detail-retention ratio from Laplacian variance inside the face region.

    Args:
        original: Reference RGB uint8 image (H, W, 3).
        generated: Modified RGB uint8 image (H, W, 3).
        mask: Binary face skin mask (H, W, uint8). Pixels > 0 are included.
              If None, the entire image is used.
        eps: Small constant to prevent division by zero when the original
             has near-zero texture (e.g., solid-colour test images).

    Returns:
        float: Detail retention ratio clamped to [0.0, 2.0].
    """
    # Convert to grayscale; Laplacian is applied to luminance channel
    orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gen_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 3×3 Laplacian kernel — standard operator for edge / texture energy
    lap_orig = cv2.Laplacian(orig_gray, cv2.CV_32F, ksize=3)
    lap_gen = cv2.Laplacian(gen_gray, cv2.CV_32F, ksize=3)

    if mask is not None and np.count_nonzero(mask > 0) > 0:
        m = mask > 0
        var_orig = float(np.var(lap_orig[m]))
        var_gen = float(np.var(lap_gen[m]))
    else:
        var_orig = float(np.var(lap_orig))
        var_gen = float(np.var(lap_gen))

    ratio = var_gen / (var_orig + eps)
    return float(np.clip(ratio, 0.0, 2.0))
