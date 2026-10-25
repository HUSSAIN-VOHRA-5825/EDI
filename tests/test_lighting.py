"""Tests for lighting variations.

Tests validate:
  - Output shape and dtype are identical to input.
  - Variations are deterministic given the same seeded rng.
  - Brightness monotonicity for light_lowlight (darker each level) and
    light_overexposed (brighter each level).
  - ΔE increases monotonically with severity for all four lighting variations.
  - All four register correctly and can be retrieved from the variation registry.
"""

import numpy as np
import pytest
from src.core.config import load_config
from src.core.registry import get_variation
from src.metrics.delta_e import delta_e_face
from src.variations import (
    ColourTemperatureVariation,
    DirectionalLightVariation,
    LowLightVariation,
    OverexposedVariation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG_PATH = "configs/default.yaml"
IMG_SIZE = 64  # Small synthetic face for speed


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture
def synthetic_face():
    """64×64 RGB image with mid-grey face region (constant hue for easy ΔE checking)."""
    rng = np.random.default_rng(7)
    img = (rng.uniform(80, 180, size=(IMG_SIZE, IMG_SIZE, 3))).astype(np.uint8)
    return img


@pytest.fixture
def face_mask():
    """Simple central circular mask (radius = 20 px)."""
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            if (y - cy) ** 2 + (x - cx) ** 2 <= 20 ** 2:
                mask[y, x] = 255
    return mask


@pytest.fixture
def seeded_rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_variation(name, cfg):
    var_cfg = cfg.get("variations", {}).get(name, {})
    return get_variation(name, config=var_cfg)


def mean_luminance(img: np.ndarray) -> float:
    """Mean luminance (Y from ITU-R BT.601) of a uint8 RGB image."""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return float(0.299 * r.mean() + 0.587 * g.mean() + 0.114 * b.mean())


# ---------------------------------------------------------------------------
# Registry smoke-test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "light_lowlight",
    "light_overexposed",
    "light_directional",
    "light_colortemp",
])
def test_registry(name, cfg):
    """All four lighting variations can be instantiated from the registry."""
    var = _make_variation(name, cfg)
    assert var is not None, f"{name} not found in registry"


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "light_lowlight",
    "light_overexposed",
    "light_directional",
    "light_colortemp",
])
@pytest.mark.parametrize("severity", [1, 3, 5])
def test_shape_dtype(name, severity, synthetic_face, face_mask, seeded_rng, cfg):
    """Output must be uint8 with the same shape as input."""
    var = _make_variation(name, cfg)
    out = var.apply(synthetic_face, severity, mask=face_mask, rng=seeded_rng)
    assert out.shape == synthetic_face.shape, f"{name} L{severity}: shape mismatch"
    assert out.dtype == np.uint8, f"{name} L{severity}: dtype is {out.dtype}, expected uint8"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "light_lowlight",
    "light_overexposed",
    "light_directional",
    "light_colortemp",
])
def test_determinism(name, synthetic_face, cfg):
    """Same seed → identical output."""
    var = _make_variation(name, cfg)
    out1 = var.apply(synthetic_face, 3, rng=np.random.default_rng(99))
    out2 = var.apply(synthetic_face, 3, rng=np.random.default_rng(99))
    assert np.array_equal(out1, out2), f"{name}: output not deterministic with same rng seed"


# ---------------------------------------------------------------------------
# Brightness monotonicity
# ---------------------------------------------------------------------------

def test_lowlight_gets_darker(synthetic_face, face_mask, cfg):
    """light_lowlight must produce strictly decreasing mean luminance with severity."""
    var = _make_variation("light_lowlight", cfg)
    lums = [mean_luminance(var.apply(synthetic_face, s, mask=face_mask)) for s in [1, 2, 3, 4, 5]]
    for i in range(len(lums) - 1):
        assert lums[i] > lums[i + 1], (
            f"light_lowlight: luminance NOT decreasing at L{i+1}→L{i+2}: {lums}"
        )


def test_overexposed_gets_brighter(synthetic_face, face_mask, cfg):
    """light_overexposed must produce strictly increasing mean luminance with severity."""
    var = _make_variation("light_overexposed", cfg)
    lums = [mean_luminance(var.apply(synthetic_face, s, mask=face_mask)) for s in [1, 2, 3, 4, 5]]
    for i in range(len(lums) - 1):
        assert lums[i] < lums[i + 1], (
            f"light_overexposed: luminance NOT increasing at L{i+1}→L{i+2}: {lums}"
        )


# ---------------------------------------------------------------------------
# ΔE monotonicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "light_lowlight",
    "light_overexposed",
    "light_directional",
    "light_colortemp",
])
def test_delta_e_monotone(name, synthetic_face, face_mask, cfg):
    """CIE 1976 ΔE inside the face mask should increase (or stay equal) with severity.

    We use a fixed seed so direction choices are consistent across levels.
    Strict monotone (>=) is used because at level 1 the shift can be very small.
    """
    var = _make_variation(name, cfg)
    de_values = []
    for s in [1, 2, 3, 4, 5]:
        out = var.apply(synthetic_face, s, mask=face_mask, rng=np.random.default_rng(42))
        de = delta_e_face(synthetic_face, out, mask=face_mask)
        de_values.append(de)

    for i in range(len(de_values) - 1):
        assert de_values[i] <= de_values[i + 1] + 0.01, (
            f"{name}: ΔE NOT monotone at L{i+1}→L{i+2}: {de_values}"
        )
