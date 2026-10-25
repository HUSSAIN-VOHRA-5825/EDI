"""Skin tone variations module.

Shifts facial and neck skin tone realistically in true CIELAB space to test
the fairness, bias, and robustness of medical AI systems across Fitzpatrick skin types.
Applies additive shifts inside a feathered skin edit mask (face skin + forehead + visible
neck, excluding hair, eyes, and eyebrows) while strictly preserving local skin texture
and identity.  Lips receive the same shift at 50 % strength with a feathered edge.

Four registered variations:
- skin_lighter: additive L* brightening with chroma compensation
- skin_darker: additive L* darkening with chroma compensation
- skin_warm: positive b* (yellow/golden) and subtle positive a* (erythema/redness) shift
- skin_cool: negative b* (cyan/blue-cool) shift; da* is a very small negative value to
             counteract the purple cast that a pure b* shift produces (cyanosis risk).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from skimage.color import lab2rgb, rgb2lab

from src.core.base import Variation
from src.core.registry import register_variation
from src.face.mesh import FaceMeshDetector, get_default_detector

logger = logging.getLogger(__name__)

# Global reference: used only for the edit mask, NOT for metrics
_EDIT_DETECTOR: Optional[FaceMeshDetector] = None


def _get_edit_detector() -> FaceMeshDetector:
    global _EDIT_DETECTOR
    if _EDIT_DETECTOR is None:
        _EDIT_DETECTOR = get_default_detector()
    return _EDIT_DETECTOR


class SkinToneBaseVariation(Variation):
    """Base class for CIELAB-space skin tone adjustments."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, config=config)
        self.last_out_of_gamut_fraction: Optional[float] = None

    def _get_shift(self, severity: int) -> Tuple[float, float, float, float]:
        """Return (dL, da, db, chroma_scale) for the given severity."""
        raise NotImplementedError

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Apply deterministic additive CIELAB skin tone shift inside feathered edit mask.

        Args:
            image: Original RGB image as uint8 NumPy array (H, W, 3).
            severity: Integer severity level (1 to 5).
            mask: Optional reference mask (unused here; edit mask is computed internally).
            rng: Unused (deterministic operation).

        Returns:
            np.ndarray: Modified RGB image as uint8 NumPy array (H, W, 3).
        """
        h, w = image.shape[:2]

        # Read optional forehead YCrCb threshold from config (set per-variation or globally)
        fwd_thresh = float(
            self.config.get("forehead_ycrcb_dist_threshold", 22.0)
        )

        # 1. Obtain skin edit mask (forehead + face + neck, without eyes, eyebrows, hair)
        #    and lip mask for partial blending (0c).
        detector = _get_edit_detector()
        result = detector.generate_skin_edit_mask(
            image,
            return_lip_mask=True,
            forehead_ycrcb_dist_threshold=fwd_thresh,
        )
        if isinstance(result, tuple):
            edit_mask, lip_mask = result
        else:
            edit_mask, lip_mask = result, None

        if edit_mask is None or np.count_nonzero(edit_mask) == 0:
            logger.warning(
                "No face/skin detected in image for %s; returning original image unchanged.", self.name
            )
            self.last_out_of_gamut_fraction = 0.0
            return image.copy()

        # 2. Feather edit mask with Gaussian blur (sigma ~ 0.5 % image width) for seamless alpha blending
        sigma = max(1.0, 0.005 * w)
        ksize = int(round(sigma * 4)) * 2 + 1
        alpha = cv2.GaussianBlur(
            edit_mask.astype(np.float32) / 255.0, (ksize, ksize), sigmaX=sigma, sigmaY=sigma
        )
        alpha = np.clip(alpha, 0.0, 1.0)

        # Enforce that pixels far outside edit region stay strictly unchanged
        dilated_edit = cv2.dilate(edit_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)))
        alpha[dilated_edit == 0] = 0.0

        # 3. Convert to true float CIELAB (L* in 0..100, a* and b* in ~-128..127)
        lab = rgb2lab(image.astype(np.float32) / 255.0)
        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        dL, da, db, chroma_scale = self._get_shift(severity)

        # 4. Apply additive shift inside edit region with chroma compensation
        L_new = L + dL
        a_new = (a + da) * chroma_scale
        b_new = (b + db) * chroma_scale

        lab_shifted = np.dstack([L_new, a_new, b_new])
        rgb_shifted = lab2rgb(lab_shifted) * 255.0

        # Count fraction of out-of-gamut pixels in the edit region before clipping
        in_edit = edit_mask > 0
        if np.any(in_edit):
            edit_pixels = rgb_shifted[in_edit]
            oog = np.any((edit_pixels < 0.0) | (edit_pixels > 255.0), axis=-1)
            self.last_out_of_gamut_fraction = float(np.mean(oog))
        else:
            self.last_out_of_gamut_fraction = 0.0

        # Clip back to valid sRGB gamut and blend using feathered alpha
        rgb_clipped = np.clip(rgb_shifted, 0.0, 255.0)
        alpha_3d = alpha[:, :, np.newaxis]
        blended = rgb_clipped * alpha_3d + image.astype(np.float32) * (1.0 - alpha_3d)

        result_img = np.clip(np.round(blended), 0, 255).astype(np.uint8)
        # Ensure pixels outside dilated edit mask are exactly identical to original
        result_img[dilated_edit == 0] = image[dilated_edit == 0]

        # 5. 0c: Blend lips at 50 % strength (feathered) so they do not look pasted on.
        #    The lip region was carved from the edit mask; here we re-apply the shift at half strength.
        if lip_mask is not None and np.any(lip_mask > 0):
            lip_sigma = max(1.0, 0.005 * w)
            lip_ksize = int(round(lip_sigma * 4)) * 2 + 1
            lip_alpha_f = cv2.GaussianBlur(
                lip_mask.astype(np.float32) / 255.0,
                (lip_ksize, lip_ksize),
                sigmaX=lip_sigma,
                sigmaY=lip_sigma,
            )
            lip_alpha_f = np.clip(lip_alpha_f * 0.50, 0.0, 1.0)  # 50 % blend strength
            lip_alpha_3d = lip_alpha_f[:, :, np.newaxis]
            result_img = np.clip(
                rgb_clipped * lip_alpha_3d + result_img.astype(np.float32) * (1.0 - lip_alpha_3d),
                0, 255,
            ).astype(np.uint8)

        return result_img


@register_variation("skin_lighter")
class SkinLighterVariation(SkinToneBaseVariation):
    """Lightens facial skin tone additively with chroma compensation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="skin_lighter", config=config)

    def _get_shift(self, severity: int) -> Tuple[float, float, float, float]:
        level_cfg = self.get_level_params(severity)
        dL = float(level_cfg.get("dL", [3.0, 6.0, 9.0, 13.0, 17.0][severity - 1]))
        k = float(self.config.get("chroma_k", 0.3))
        chroma_scale = 1.0 + k * (-dL) / 100.0
        return dL, 0.0, 0.0, chroma_scale


@register_variation("skin_darker")
class SkinDarkerVariation(SkinToneBaseVariation):
    """Darkens facial skin tone additively with chroma compensation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="skin_darker", config=config)

    def _get_shift(self, severity: int) -> Tuple[float, float, float, float]:
        level_cfg = self.get_level_params(severity)
        dL = float(level_cfg.get("dL", [-3.5, -7.0, -10.5, -14.0, -17.5][severity - 1]))
        k = float(self.config.get("chroma_k", 0.3))
        chroma_scale = 1.0 + k * (-dL) / 100.0
        return dL, 0.0, 0.0, chroma_scale


@register_variation("skin_warm")
class SkinWarmVariation(SkinToneBaseVariation):
    """Shifts skin tone toward warmer (golden/erythematous) spectrum."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="skin_warm", config=config)

    def _get_shift(self, severity: int) -> Tuple[float, float, float, float]:
        level_cfg = self.get_level_params(severity)
        db = float(level_cfg.get("db", [2.0, 4.0, 6.0, 9.0, 12.0][severity - 1]))
        da = float(level_cfg.get("da", [0.5, 1.0, 1.5, 2.0, 3.0][severity - 1]))
        return 0.0, da, db, 1.0


@register_variation("skin_cool")
class SkinCoolVariation(SkinToneBaseVariation):
    """Shifts skin tone toward cooler (cyan/blue undertone) spectrum.

    da* is read from config and is a small negative value (≈ −0.3 to −1.8) that
    counteracts the purple cast a pure negative b* shift would produce — which
    could be mistaken for cyanosis in clinical review.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="skin_cool", config=config)

    def _get_shift(self, severity: int) -> Tuple[float, float, float, float]:
        level_cfg = self.get_level_params(severity)
        db = float(level_cfg.get("db", [-1.5, -3.0, -4.5, -6.0, -8.0][severity - 1]))
        da = float(level_cfg.get("da", [-0.3, -0.6, -1.0, -1.4, -1.8][severity - 1]))
        return 0.0, da, db, 1.0
