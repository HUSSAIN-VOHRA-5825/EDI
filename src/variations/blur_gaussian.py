"""Gaussian blur variation.

Simulates optical defocus or atmospheric MTF degradation using an isotropic Gaussian filter.
Parameters are defined as a fraction of image width so the visual degradation scale
remains constant across arbitrary resolutions (e.g., 512px vs 4K).
"""

from typing import Any, Dict, Optional
import cv2
import numpy as np
from src.core.base import Variation
from src.core.registry import register_variation


@register_variation("blur_gaussian")
class GaussianBlurVariation(Variation):
    """Applies isotropic Gaussian blur scaled proportionally to image width."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="blur_gaussian", config=config)

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        params = self.get_level_params(severity)
        # Sigma is specified as a fraction of image width (resolution-invariant)
        sigma_fraction = float(params.get("sigma_fraction", 0.002 * severity))

        h, w = image.shape[:2]
        sigma = max(0.1, sigma_fraction * w)

        # Work in float32 for high numerical precision
        img_float = image.astype(np.float32)

        # Reflect border padding prevents boundary darkening artifacts
        # Passing ksize=(0, 0) lets OpenCV compute the appropriate odd kernel size from sigma
        blurred = cv2.GaussianBlur(
            img_float,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )

        return np.clip(blurred, 0.0, 255.0).astype(np.uint8)
