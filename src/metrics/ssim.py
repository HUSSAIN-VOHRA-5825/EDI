"""Structural Similarity Index Measure (SSIM) metric.

Quantifies luminance, contrast, and structural degradation between the original
and generated images, both across the entire canvas and specifically inside the
clinically relevant facial skin mask.
"""

import logging
from typing import Optional
import cv2
import numpy as np
from skimage.metrics import structural_similarity

logger = logging.getLogger(__name__)


def ssim_score(
    img_a: np.ndarray,
    img_b: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Calculate Structural Similarity Index (SSIM) between two images.

    Args:
        img_a: Reference RGB image as uint8 NumPy array (H, W, 3).
        img_b: Generated RGB image as uint8 NumPy array (H, W, 3).
        mask: Optional binary uint8 mask (H, W). If provided, SSIM is averaged
            exclusively over pixels where mask > 0.

    Returns:
        float: SSIM score in range [-1.0, 1.0] (1.0 indicates perfect identity).
    """
    # Ensure identical spatial dimensions for comparison
    if img_a.shape[:2] != img_b.shape[:2]:
        logger.warning(
            "SSIM: Dimensions mismatch (%s vs %s). Resizing img_b to match img_a.",
            img_a.shape[:2],
            img_b.shape[:2],
        )
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Determine dynamic range for uint8
    data_range = 255.0

    # Case 1: Masked SSIM restricted to region of interest (e.g. facial skin)
    if mask is not None:
        if mask.shape[:2] != img_a.shape[:2]:
            mask = cv2.resize(mask, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask_pixels = np.count_nonzero(mask > 0)
        if mask_pixels == 0:
            logger.warning("SSIM: Supplied mask has 0 positive pixels. Falling back to full image SSIM.")
            score = structural_similarity(img_a, img_b, channel_axis=2, data_range=data_range)
            return float(score)

        # Compute full pixel-level SSIM response map
        _, ssim_map = structural_similarity(
            img_a,
            img_b,
            channel_axis=2,
            data_range=data_range,
            full=True,
        )

        # Average SSIM map across the masked skin pixels and across color channels
        # ssim_map has shape (H, W, 3)
        masked_region = mask > 0
        score = float(np.mean(ssim_map[masked_region]))
        return score

    # Case 2: Full-frame SSIM
    score = structural_similarity(
        img_a,
        img_b,
        channel_axis=2,
        data_range=data_range,
    )
    return float(score)
