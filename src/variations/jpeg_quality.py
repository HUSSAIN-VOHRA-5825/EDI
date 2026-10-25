"""JPEG compression quality variation.

Simulates lossy DCT block-quantization artifacts, chroma subsampling (4:2:0), and high-frequency
ringing typical of compressed telemedicine and web-transmitted facial images.
Operates entirely in-memory using cv2.imencode/imdecode without temporary filesystem I/O.
"""

from typing import Any, Dict, Optional
import cv2
import numpy as np
from src.core.base import Variation
from src.core.registry import register_variation


@register_variation("jpeg_quality")
class JpegQualityVariation(Variation):
    """Applies JPEG compression artifacts with configurable quality factor (1-100)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="jpeg_quality", config=config)

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        params = self.get_level_params(severity)
        quality = int(params.get("quality", max(5, 80 - severity * 15)))
        quality = max(1, min(100, quality))

        # OpenCV expects BGR color order for encoding
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # In-memory JPEG encoding into memory buffer
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", img_bgr, encode_params)
        if not success or buffer is None:
            raise RuntimeError(f"JPEG encoding failed at quality={quality}")

        # In-memory JPEG decoding
        decoded_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            raise RuntimeError(f"JPEG decoding failed for buffer of size {len(buffer)}")

        # Convert back to RGB uint8
        return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
