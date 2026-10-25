"""Low-resolution JPEG upload variation.

Simulates mobile or web telehealth portals where images are downscaled, compressed,
and subsequently upsampled by clinical viewer software, introducing combined pixelation,
quantization, and interpolation blur.
"""

from typing import Any, Dict, Optional
import cv2
import numpy as np
from src.core.base import Variation
from src.core.registry import register_variation


@register_variation("jpeg_lowres")
class JpegLowResVariation(Variation):
    """Simulates downscaled upload with JPEG compression and bilinear upsampling."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="jpeg_lowres", config=config)

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        params = self.get_level_params(severity)
        scale_factor = float(params.get("scale_factor", max(0.2, 1.0 - 0.15 * severity)))
        quality = int(params.get("quality", max(10, 95 - 15 * severity)))

        h, w = image.shape[:2]
        down_w = max(1, int(round(w * scale_factor)))
        down_h = max(1, int(round(h * scale_factor)))

        # 1. Downscale to simulated low-resolution upload size (using area resampling)
        img_down = cv2.resize(image, (down_w, down_h), interpolation=cv2.INTER_AREA)

        # 2. Apply in-memory JPEG compression
        img_down_bgr = cv2.cvtColor(img_down, cv2.COLOR_RGB2BGR)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, quality))]
        success, buffer = cv2.imencode(".jpg", img_down_bgr, encode_params)
        if not success or buffer is None:
            raise RuntimeError(f"JPEG encoding failed in lowres variation at quality={quality}")

        decoded_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            raise RuntimeError("JPEG decoding failed in lowres variation")
        img_jpeg_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

        # 3. Bilinear upsampling back to original spatial canvas size
        img_upscaled = cv2.resize(img_jpeg_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        return np.clip(img_upscaled, 0, 255).astype(np.uint8)
