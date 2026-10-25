"""Test-only variations for pipeline verification and end-to-end testing.

NOTE: These are dummy variations strictly intended to prove that the pipeline,
registry, config loading, and metrics evaluation work properly end-to-end.
Real variations (blur, noise, etc.) are built sequentially in subsequent tasks.
"""

from typing import Any, Dict, Optional
import numpy as np
from src.core.base import Variation
from src.core.registry import register_variation


@register_variation("identity_test")
class IdentityTestVariation(Variation):
    """Test variation that returns the original image unchanged.

    Used to verify:
    1. Pipeline end-to-end execution.
    2. Identity similarity benchmark (should score ~1.0).
    3. Full and masked SSIM benchmarks (should score 1.0).
    4. LPIPS benchmark (should score 0.0).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="identity_test", config=config)

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        # Validate severity level range
        _ = self.get_level_params(severity)
        # Returns an unmodified copy of the input image
        return image.copy()


@register_variation("brightness_test")
class BrightnessTestVariation(Variation):
    """Test variation that scales image brightness according to severity.

    Used to verify:
    1. Configuration-driven parameter scaling per severity level.
    2. Monotonic degradation of structural and perceptual similarity metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="brightness_test", config=config)

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        params = self.get_level_params(severity)
        # Read factor from config (defaulting to 1.0 + severity * 0.15 if omitted)
        factor = float(params.get("factor", 1.0 + severity * 0.15))

        # Multiply pixel intensities in floating point and clip to uint8 [0, 255]
        brightened = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return brightened
