"""Realistic camera sensor noise variation.

Physical CMOS image sensors capture photons according to Poisson statistics (quantum shot noise),
compounded by electronic read/amplifier noise before gamma encoding.
This module:
1. Converts gamma-encoded sRGB to linear radiance space (gamma 2.2 un-mapping).
2. Applies Poisson photon arrival statistics (shot noise) scaled by photon count (simulating ISO gain).
3. Adds Gaussian read noise from analog-to-digital sensor conversion.
4. Converts back to gamma space, resulting in noise that is physically stronger in darker regions.
"""

from typing import Any, Dict, Optional
import numpy as np
from src.core.base import Variation
from src.core.registry import register_variation


@register_variation("noise_sensor")
class SensorNoiseVariation(Variation):
    """Simulates physical camera sensor noise (Poisson shot + Gaussian read in linear light)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="noise_sensor", config=config)

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
        # photon_scale: simulated full-well electron capacity / ISO gain (lower = higher ISO / higher noise)
        photon_scale = float(params.get("photon_scale", 4000.0 / severity))
        # read_noise_std: standard deviation of electronic read noise in linear [0, 1] units
        read_noise_std = float(params.get("read_noise_std", 0.005 * severity))

        # 1. Convert from non-linear gamma 2.2 sRGB display space to linear photon flux [0.0, 1.0]
        img_norm = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)
        linear = np.power(img_norm, 2.2)

        # 2. Simulate Poisson quantum shot noise on photon counts
        expected_photons = np.clip(linear * photon_scale, 0.0, None)
        # Sample Poisson count and normalize back to linear scale
        shot_electrons = rng.poisson(expected_photons).astype(np.float32)
        shot_linear = shot_electrons / max(1.0, photon_scale)

        # 3. Additive Gaussian sensor read/thermal noise
        read_noise = rng.normal(loc=0.0, scale=read_noise_std, size=image.shape).astype(np.float32)

        total_linear = np.clip(shot_linear + read_noise, 0.0, 1.0)

        # 4. Re-apply display gamma 2.2 encoding
        gamma_corrected = np.power(total_linear, 1.0 / 2.2) * 255.0

        return np.clip(gamma_corrected, 0.0, 255.0).astype(np.uint8)
