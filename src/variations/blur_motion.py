"""Motion blur variation — 4× supersampled kernel build.

Simulates camera shake or patient movement during clinical photography.
Builds a motion PSF by:
1. Drawing the line at 4× the target kernel resolution (supersampled),
2. Downsampling with INTER_AREA (box filter) so fractional-pixel gaps
   in the diagonal are smoothed out and the line is continuous,
3. Normalising to sum = 1 (energy-conserving).

Parameters are defined as a fraction of image width so the visual
trajectory remains consistent across arbitrary resolutions.
"""

from typing import Any, Dict, Optional
import cv2
import numpy as np
from src.core.base import Variation
from src.core.registry import register_variation


@register_variation("blur_motion")
class MotionBlurVariation(Variation):
    """Applies directional motion blur with a 4× supersampled anti-aliased line kernel."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="blur_motion", config=config)

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
        # Length is specified as a fraction of image width (resolution-invariant)
        length_fraction = float(params.get("length_fraction", 0.008 * severity))

        h, w = image.shape[:2]
        length_px = max(1.0, length_fraction * w)

        # Draw deterministic random orientation angle derived from base seed + image + step
        from src.utils.seeding import get_direction_rng
        dir_rng = get_direction_rng(image, "blur_motion", rng=rng, config_seed=self.config.get("seed"), instance=self)
        angle_deg = float(dir_rng.uniform(0.0, 180.0))
        angle_rad = np.radians(angle_deg)
        self.last_angle_deg = angle_deg

        # Final kernel is odd-sized and large enough to contain the motion arc
        ksize = max(3, int(np.ceil(length_px)) | 1)

        # --- 4× supersampled kernel build ---
        # Drawing at 4× scale means each output pixel maps to a 4×4 block,
        # so diagonal steps never leave gaps; INTER_AREA then integrates each block.
        scale = 4
        high_ksize = ksize * scale
        high_kernel = np.zeros((high_ksize, high_ksize), dtype=np.float32)

        cx = (high_ksize - 1) / 2.0
        cy = (high_ksize - 1) / 2.0
        half_len = (length_px * scale) / 2.0
        dx = half_len * np.cos(angle_rad)
        dy = half_len * np.sin(angle_rad)

        pt1 = (int(round(cx - dx)), int(round(cy - dy)))
        pt2 = (int(round(cx + dx)), int(round(cy + dy)))

        # Draw at supersampled scale; thickness=scale fills the 4×4 blocks
        cv2.line(high_kernel, pt1, pt2, 1.0, thickness=scale, lineType=cv2.LINE_AA)

        # Box-filter downsample back to target kernel size
        kernel = cv2.resize(high_kernel, (ksize, ksize), interpolation=cv2.INTER_AREA)

        # Normalise so that the sum equals exactly 1 (brightness-conserving)
        k_sum = kernel.sum()
        if k_sum > 0:
            kernel /= k_sum
        else:
            kernel[ksize // 2, ksize // 2] = 1.0

        img_float = image.astype(np.float32)
        # BORDER_REFLECT eliminates dark-edge vignetting at image boundaries
        blurred = cv2.filter2D(img_float, -1, kernel, borderType=cv2.BORDER_REFLECT)

        return np.clip(blurred, 0.0, 255.0).astype(np.uint8)
