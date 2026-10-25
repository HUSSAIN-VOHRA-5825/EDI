"""Unit tests for the 7 core image variations (Blur, Noise, JPEG).

Validates:
1. Output matches input shape and uint8 dtype.
2. Determinism: identical random seed produces identical pixel output.
3. Execution on small (64x64) inputs without crashing or boundary exceptions.
4. Monotonicity: distortion (1 - SSIM against original) is strictly increasing across levels 1 to 5.
5. Photometric conservation: brightness (mean pixel intensity) is preserved within 1.0 unit for blur variations.
"""

from pathlib import Path
import numpy as np
import pytest
from src.core.config import load_config
from src.core.registry import get_variation
from src.metrics.ssim import ssim_score
from src.utils.io import load_image
from src.utils.seeding import get_rng

ALL_VARIATION_NAMES = [
    "blur_gaussian",
    "blur_motion",
    "blur_defocus",
    "noise_gaussian",
    "noise_sensor",
    "jpeg_quality",
    "jpeg_lowres",
]

BLUR_VARIATION_NAMES = [
    "blur_gaussian",
    "blur_motion",
    "blur_defocus",
]


@pytest.fixture(scope="module")
def config():
    """Load default YAML config."""
    return load_config("configs/default.yaml")


@pytest.fixture(scope="module")
def test_face_image():
    """Load reference face image."""
    p = Path("data/input/sample_face.png")
    if p.is_file():
        return load_image(p)
    import skimage.data
    return skimage.data.astronaut()


@pytest.mark.parametrize("var_name", ALL_VARIATION_NAMES)
def test_variation_shape_dtype_and_determinism(var_name: str, config, test_face_image: np.ndarray):
    """Verify output shape, dtype, and deterministic repeatability with same seed."""
    var = get_variation(var_name, config=config["variations"][var_name])

    out1 = var.apply(test_face_image, severity=3, rng=get_rng(12345))
    out2 = var.apply(test_face_image, severity=3, rng=get_rng(12345))

    assert out1.shape == test_face_image.shape, f"{var_name}: Shape changed."
    assert out1.dtype == np.uint8, f"{var_name}: Dtype is not uint8."
    assert np.array_equal(out1, out2), f"{var_name}: Determinism failed with identical seed."


@pytest.mark.parametrize("var_name", ALL_VARIATION_NAMES)
def test_variation_on_small_64x64_image(var_name: str, config):
    """Verify variation handles tiny 64x64 resolution without shape or kernel border errors."""
    tiny_img = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    var = get_variation(var_name, config=config["variations"][var_name])

    out = var.apply(tiny_img, severity=5, rng=get_rng(42))
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("var_name", ALL_VARIATION_NAMES)
def test_variation_monotonic_distortion(var_name: str, config, test_face_image: np.ndarray):
    """Verify distortion (1 - SSIM against original) strictly increases from level 1 to 5."""
    var = get_variation(var_name, config=config["variations"][var_name])

    distortions = []
    for lvl in range(1, 6):
        out = var.apply(test_face_image, severity=lvl, rng=get_rng(42))
        dist = 1.0 - ssim_score(test_face_image, out)
        distortions.append(dist)

    for i in range(len(distortions) - 1):
        assert distortions[i] < distortions[i + 1], (
            f"{var_name}: Distortion not strictly increasing at level {i+1}->{i+2}: "
            f"{distortions[i]:.4f} >= {distortions[i+1]:.4f}"
        )


@pytest.mark.parametrize("var_name", BLUR_VARIATION_NAMES)
def test_blur_preserves_overall_brightness(var_name: str, config, test_face_image: np.ndarray):
    """Verify blur variations conserve photometric brightness (mean intensity within 1.0 unit)."""
    var = get_variation(var_name, config=config["variations"][var_name])
    orig_mean = float(test_face_image.mean())

    for lvl in range(1, 6):
        out = var.apply(test_face_image, severity=lvl, rng=get_rng(42))
        out_mean = float(out.mean())
        assert abs(out_mean - orig_mean) < 1.0, (
            f"{var_name}: Level {lvl} mean intensity shifted by {abs(out_mean - orig_mean):.2f}"
        )
