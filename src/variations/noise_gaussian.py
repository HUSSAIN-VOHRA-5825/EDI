"""Additive Gaussian noise variation.

Simulates thermal electronic noise, amplifier distortion, or transmission interference
as independent, identically distributed zero-mean Gaussian perturbations across pixel channels.
"""

from typing import Any, Dict, Optional
import numpy as np
from src.core.base import Variation
from src.core.registry import register_variation


@register_variation("noise_gaussian")
class GaussianNoiseVariation(Variation):
    """Applies additive zero-mean Gaussian noise scaled by configured sigma."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="noise_gaussian", config=config)

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(42)

        params = self.get_level_params(severity)
        # Sigma is specified on a 0-255 scale
        sigma = float(params.get("sigma", 4.0 * severity))

        # Sample Gaussian perturbations using the seeded isolated generator
        noise = rng.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)

        # Work in float32, add noise, clip to valid dynamic range, and return uint8
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0.0, 255.0).astype(np.uint8)
