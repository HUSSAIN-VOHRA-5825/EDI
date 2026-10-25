"""Defocus blur variation.

Simulates shallow depth-of-field or optical lens misfocus by convolving with a circular
disk kernel (pillbox Point Spread Function / bokeh).
Parameters are defined as a fraction of image width to remain resolution-independent.
"""

from typing import Any, Dict, Optional
import cv2
import numpy as np
from src.core.base import Variation
from src.core.registry import register_variation


@register_variation("blur_defocus")
class DefocusBlurVariation(Variation):
    """Applies optical defocus blur with a normalized circular disk kernel."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="blur_defocus", config=config)

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        params = self.get_level_params(severity)
        # Radius is specified as a fraction of image width (resolution-invariant)
        radius_fraction = float(params.get("radius_fraction", 0.006 * severity))

        h, w = image.shape[:2]
        radius_px = max(0.5, radius_fraction * w)

        # Construct an odd-sized kernel grid enclosing the disk
        ksize = max(3, int(np.ceil(radius_px * 2.0)) | 1)
        center = (ksize - 1) / 2.0

        y, x = np.ogrid[:ksize, :ksize]
        dist_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2)

        # Anti-aliased disk kernel: 1.0 inside, subpixel linear falloff in boundary ring
        kernel = np.clip(radius_px + 0.5 - dist_from_center, 0.0, 1.0).astype(np.float32)

        # Normalize kernel sum to 1.0 to preserve total image brightness
        k_sum = kernel.sum()
        if k_sum > 0:
            kernel /= k_sum
        else:
            kernel[ksize // 2, ksize // 2] = 1.0

        img_float = image.astype(np.float32)
        # BORDER_REFLECT avoids dark borders around image edges
        blurred = cv2.filter2D(img_float, -1, kernel, borderType=cv2.BORDER_REFLECT)

        return np.clip(blurred, 0.0, 255.0).astype(np.uint8)
