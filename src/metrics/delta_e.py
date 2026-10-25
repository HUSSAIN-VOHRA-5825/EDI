"""CIE 1976 ΔE colour-difference metric inside the face region.

ΔE (Delta-E) measures perceptual colour change in CIE L*a*b* space.
It is used here as an INFORMATIONAL metric only — it does NOT affect
the pass/fail gate — because colour shifts (from lighting and skin-tone
variations) are intentional perturbations, not failures.

However, it quantifies HOW MUCH colour has changed, which is useful for:
  - Calibrating severity levels for lighting / skin-tone variations.
  - Flagging unintended colour shifts in nominally achromatic variations
    (blur, noise, jpeg) — a non-zero ΔE in those indicates a bug.

CIE 1976 formula (L2 distance in Lab): ΔE = sqrt((ΔL)² + (Δa)² + (Δb)²)

We report the MEAN ΔE across all pixels inside the face mask.
Values ≤ 1.0 are imperceptible; ≥ 10 correspond to very obvious shifts.
"""

from typing import Optional
import cv2
import numpy as np


def delta_e_face(
    original: np.ndarray,
    generated: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute mean CIE 1976 ΔE inside the face skin mask.

    Args:
        original: Reference RGB uint8 image (H, W, 3).
        generated: Modified RGB uint8 image (H, W, 3).
        mask: Binary face skin mask (H, W, uint8). If None, uses entire image.

    Returns:
        float: Mean CIE 1976 ΔE across face-region pixels, rounded to 4 d.p.
    """
    # Convert to CIE L*a*b* (OpenCV expects BGR input)
    orig_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    gen_bgr = cv2.cvtColor(generated, cv2.COLOR_RGB2BGR)

    orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    gen_lab = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

    # OpenCV stores Lab as L: 0-100 scaled to 0-255, a/b: -127..127 offset to 0-255.
    # Rescale to true Lab range for a valid Euclidean distance.
    orig_lab[:, :, 0] *= 100.0 / 255.0          # L*: [0, 100]
    orig_lab[:, :, 1] -= 128.0                   # a*: [-128, 127]
    orig_lab[:, :, 2] -= 128.0                   # b*: [-128, 127]
    gen_lab[:, :, 0] *= 100.0 / 255.0
    gen_lab[:, :, 1] -= 128.0
    gen_lab[:, :, 2] -= 128.0

    # Euclidean distance per pixel in Lab space = CIE 1976 ΔE
    diff = orig_lab - gen_lab
    de_map = np.sqrt(np.sum(diff ** 2, axis=-1))  # (H, W)

    if mask is not None and np.count_nonzero(mask > 0) > 0:
        mean_de = float(np.mean(de_map[mask > 0]))
    else:
        mean_de = float(np.mean(de_map))

    return round(mean_de, 4)
