"""Lighting variation implementations.

Four registered variations that simulate clinical imaging lighting artefacts.
All operate in **linear light** (sRGB gamma removed → effect applied → gamma restored)
to match physical exposure physics.  Effects are applied to the WHOLE image;
the face mask is used ONLY for metric calculation.

sRGB ↔ linear conversion follows IEC 61966-2-1:
  linear = ((srgb + 0.055) / 1.055) ** 2.4  for srgb > 0.04045
         =  srgb / 12.92                      otherwise
  srgb   = 1.055 * linear ** (1/2.4) - 0.055 for linear > 0.0031308
         = 12.92 * linear                     otherwise

All severity parameters come from configs/default.yaml.
All randomness uses the seeded rng passed in at apply()-time.
Parameters are physically motivated (gains, fractions) — not fixed pixels.
"""

from typing import Any, Dict, Optional
import cv2
import numpy as np
from src.core.base import Variation
from src.core.registry import register_variation


# ---------------------------------------------------------------------------
# sRGB ↔ linear colour space helpers
# ---------------------------------------------------------------------------

def _srgb_to_linear(img_f32: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear light [0,1] using the IEC 61966-2-1 piecewise formula."""
    out = np.where(
        img_f32 <= 0.04045,
        img_f32 / 12.92,
        ((img_f32 + 0.055) / 1.055) ** 2.4,
    )
    return out.astype(np.float32)


def _linear_to_srgb(lin_f32: np.ndarray) -> np.ndarray:
    """Convert linear light [0,1] back to sRGB [0,1] using the IEC 61966-2-1 piecewise formula."""
    lin_f32 = np.clip(lin_f32, 0.0, 1.0)
    out = np.where(
        lin_f32 <= 0.0031308,
        lin_f32 * 12.92,
        1.055 * (lin_f32 ** (1.0 / 2.4)) - 0.055,
    )
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Low-light  (underexposure)
# ---------------------------------------------------------------------------

@register_variation("light_lowlight")
class LowLightVariation(Variation):
    """Simulates dim ambient lighting / sensor underexposure.

    Multiplies linear-space luminance by a gain < 1, then re-applies sRGB gamma.
    This reproduces the characteristic shadow-region noise floor and low contrast
    of a camera with ISO too low or shutter speed too fast in dim conditions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="light_lowlight", config=config)

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        params = self.get_level_params(severity)
        gain = float(params.get("gain", 0.5))

        img_f32 = image.astype(np.float32) / 255.0
        lin = _srgb_to_linear(img_f32)
        # Uniform gain across all channels — pure exposure reduction
        darkened = lin * gain
        srgb_out = _linear_to_srgb(darkened)
        return np.clip(srgb_out * 255.0, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Overexposed  (flash / sensor saturation with tanh shoulder)
# ---------------------------------------------------------------------------

@register_variation("light_overexposed")
class OverexposedVariation(Variation):
    """Simulates flash overexposure with a tanh highlight roll-off.

    A hard clip at 1.0 (the naive approach) creates an abrupt step in the
    highlight gradient that is unrealistic.  Using a tanh curve allows smooth
    shoulder compression — highlights gently roll into white rather than
    snapping, which matches actual camera highlight reconstruction behaviour.

    The tanh stretch: srgb_bright = tanh(linear * gain) normalised to [0, 1].
    At low gain values tanh ≈ linear, so mild severities show minimal distortion.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="light_overexposed", config=config)

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        params = self.get_level_params(severity)
        gain = float(params.get("gain", 2.0))

        img_f32 = image.astype(np.float32) / 255.0
        lin = _srgb_to_linear(img_f32)

        # tanh shoulder: maps linear * gain → smooth saturation
        # Normalise by tanh(1) so that at gain=1 output is unmodified.
        # The normalisation constant ensures gain=1.0 is a no-op.
        brightened = np.tanh(lin * gain) / np.tanh(1.0 * gain)

        srgb_out = _linear_to_srgb(brightened)
        return np.clip(srgb_out * 255.0, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Directional lighting  (one-sided shadow gradient)
# ---------------------------------------------------------------------------

def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Cubic Hermite smoothstep — maps [0,1] to [0,1] with zero-derivative endpoints.

    Used to create a soft gradient between the lit and shadowed halves of the face,
    avoiding the visible banding of a linear ramp.
    """
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@register_variation("light_directional")
class DirectionalLightVariation(Variation):
    """Simulates off-axis clinical ring-light casting a one-sided shadow.

    Generates a directional gradient mask:
      - A random axis (angle θ) is drawn from the seeded rng.
      - The projection of each pixel along that axis gives a signed distance.
      - A smoothstep function turns the signed distance into a soft [0,1] weight.
      - The shadow_strength parameter controls how dark the shadow side gets.
      - Effect is applied in linear space so gamma is respected.

    The gradient is anchored at the image centre so the middle of the face
    always receives approximately average illumination, matching the clinical
    scenario where the camera is aimed at the subject.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="light_directional", config=config)

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
        shadow_strength = float(params.get("shadow_strength", 0.5))

        # Random direction (angle) for the shadow axis derived from base seed + image + step
        from src.utils.seeding import get_direction_rng
        dir_rng = get_direction_rng(image, "light_directional", rng=rng, config_seed=self.config.get("seed"), instance=self)
        angle = float(dir_rng.uniform(0.0, 2.0 * np.pi))
        self.last_angle = angle
        self.last_angle_deg = float(np.degrees(angle))
        dx, dy = np.cos(angle), np.sin(angle)

        h, w = image.shape[:2]

        # Pixel coordinate grid relative to image centre (range roughly -0.5..+0.5)
        ys = (np.arange(h) - h / 2.0) / max(h, w)
        xs = (np.arange(w) - w / 2.0) / max(h, w)
        xv, yv = np.meshgrid(xs, ys)

        # Signed projection along the random axis; normalise to [-1, 1] range
        proj = (xv * dx + yv * dy)
        proj_norm = proj / (np.max(np.abs(proj)) + 1e-8)  # → [-1, 1]

        # Remap to [0, 1]: the "shadow side" maps to 0, the lit side to 1
        t = (proj_norm + 1.0) / 2.0          # [0, 1]
        gradient = _smoothstep(t)             # smooth curve: shadowed=0, lit=1

        # Shadow multiplier: shadow side = (1 - shadow_strength), lit side = 1.0
        gain_map = 1.0 - shadow_strength * (1.0 - gradient)   # (H, W)
        gain_map = gain_map[:, :, np.newaxis]                   # broadcast to (H, W, 1)

        img_f32 = image.astype(np.float32) / 255.0
        lin = _srgb_to_linear(img_f32)
        lit = np.clip(lin * gain_map, 0.0, 1.0)
        srgb_out = _linear_to_srgb(lit.astype(np.float32))
        return np.clip(srgb_out * 255.0, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Colour temperature  (warm / cool white-balance error)
# ---------------------------------------------------------------------------

# D65 white-point colour temperature matrix coefficients (simplified):
# Warm shift = increase R, decrease B (low Kelvin, incandescent / candlelight).
# Cool shift = increase B, decrease R (high Kelvin, overcast sky / LED).
# Shifts are applied as linear gains per channel in linear light.

# Per-channel gain tables for warm and cool directions:
#   warm_gains = [R_up, G_neutral, B_down]
#   cool_gains = [R_down, G_neutral, B_up]
# The R/B gains are ±shift, and G has a compensating ±(shift/3) to keep luminance
# approximately constant (luminance = 0.2126 R + 0.7152 G + 0.0722 B).

@register_variation("light_colortemp")
class ColourTemperatureVariation(Variation):
    """Simulates white-balance error / colour temperature mismatch.

    In clinical photography, mixed ambient light sources or incorrect white-balance
    settings cause a systematic warm (orange) or cool (blue) colour cast.

    Effect:
      - Direction (warm vs cool) is chosen randomly per image via the seeded rng.
      - Warm: R channel boosted, B reduced; compensated G to maintain luminance.
      - Cool: B channel boosted, R reduced; compensated G to maintain luminance.
      - Applied in linear sRGB space; sRGB gamma restored after.
      - Luminance normalisation ensures overall brightness is unchanged.

    The shift parameter is a per-channel gain offset in linear space (resolution-invariant).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="light_colortemp", config=config)

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
        shift = float(params.get("shift", 0.2))
        # Randomly choose warm (+1) or cool (-1) direction derived from base seed + image + step
        from src.utils.seeding import get_direction_rng
        dir_rng = get_direction_rng(image, "light_colortemp", rng=rng, config_seed=self.config.get("seed"), instance=self)
        direction = 1 if dir_rng.random() > 0.5 else -1
        self.last_direction = direction
        self.last_direction_str = "warm (+1)" if direction > 0 else "cool (-1)"

        # Channel gain vector in linear space
        # R = +shift (warm) or -shift (cool); B is opposite; G compensates by shift/3
        # This keeps approximate luminance constant: Δlum ≈ 0.2126*shift - 0.0722*shift - 0.7152*shift/3 ≈ 0
        r_gain = 1.0 + direction * shift
        g_gain = 1.0 - direction * (shift / 3.0)
        b_gain = 1.0 - direction * shift
        channel_gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32)

        img_f32 = image.astype(np.float32) / 255.0
        lin = _srgb_to_linear(img_f32)

        # Apply per-channel gain; clip to keep in valid range
        shifted = np.clip(lin * channel_gains[np.newaxis, np.newaxis, :], 0.0, 1.0)
        srgb_out = _linear_to_srgb(shifted.astype(np.float32))
        return np.clip(srgb_out * 255.0, 0.0, 255.0).astype(np.uint8)
