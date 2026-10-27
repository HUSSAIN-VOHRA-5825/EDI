"""tests/test_chain.py

Unit tests for chained variations:
- Deterministic output with identical seed
- Canonical ordering enforcement regardless of input order
- Output distinct from single-step variations at severity 5
- Notes recording executed steps and levels
- Skip_below behavior at severity 1
- Robustness on small images (64x64) without crashes
"""

import numpy as np
import pytest

from src.core.registry import get_variation
from src.variations.chain import ChainVariation, sort_canonical_steps


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a reproducible 128x128 synthetic face-like RGB image."""
    rng = np.random.default_rng(1234)
    img = np.full((128, 128, 3), 160, dtype=np.uint8)
    # Add some variation in tone
    noise = rng.integers(-20, 20, size=(128, 128, 3))
    return np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)


def test_chain_deterministic_with_seed(sample_image: np.ndarray) -> None:
    """Same seed gives identical output across runs."""
    chain = get_variation("chain_dim_clinic_phone")
    rng1 = np.random.default_rng(42)
    out1 = chain.apply(sample_image, severity=3, rng=rng1)

    rng2 = np.random.default_rng(42)
    out2 = chain.apply(sample_image, severity=3, rng=rng2)

    assert np.array_equal(out1, out2), "Chained variation output must be deterministic with identical seed"


def test_chain_random_deterministic_with_seed(sample_image: np.ndarray) -> None:
    """Random chain variation produces identical selections and output with same seed and image."""
    chain1 = get_variation("chain_random")
    rng1 = np.random.default_rng(999)
    out1 = chain1.apply(sample_image, severity=3, rng=rng1)
    notes1 = chain1.last_notes

    chain2 = get_variation("chain_random")
    rng2 = np.random.default_rng(999)
    out2 = chain2.apply(sample_image, severity=3, rng=rng2)
    notes2 = chain2.last_notes

    assert notes1 == notes2, "Random chain should select identical steps with same seed and image"
    assert np.array_equal(out1, out2), "Random chain output must match for identical seed"


def test_chain_random_same_recipe_across_severities(sample_image: np.ndarray) -> None:
    """Random chain keeps the exact same variation recipe across severities for the same image."""
    chain = get_variation("chain_random")
    rng = np.random.default_rng(42)

    recipe_var_names = []
    for s in range(1, 6):
        chain.apply(sample_image, severity=s, rng=rng)
        notes = chain.last_notes or ""
        # Extract base variation names without @level
        step_names = [token.split("@")[0] for token in notes.split(" > ")]
        recipe_var_names.append(step_names)

    # All severities must have the exact same list of variations
    first_recipe = recipe_var_names[0]
    for idx, r in enumerate(recipe_var_names[1:], start=2):
        assert r == first_recipe, f"Severity {idx} recipe {r} differs from severity 1 recipe {first_recipe}"


def test_dim_clinic_phone_monotonic_luminance(sample_image: np.ndarray) -> None:
    """Mean luminance decreases strictly monotonically across severities 1 to 5."""
    chain = get_variation("chain_dim_clinic_phone")
    lums = []
    for s in range(1, 6):
        out = chain.apply(sample_image, severity=s, rng=np.random.default_rng(42))
        lum = float(np.mean(0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]))
        lums.append(lum)
        assert f"light_lowlight@{s}" in (chain.last_notes or "")

    for i in range(len(lums) - 1):
        assert lums[i] > lums[i + 1], f"Luminance did not decrease: level {i+1} ({lums[i]:.2f}) <= level {i+2} ({lums[i+1]:.2f})"


def test_harsh_exam_lamp_directional_level(sample_image: np.ndarray) -> None:
    """Harsh exam lamp applies light_directional at the same level as severity."""
    chain = get_variation("chain_harsh_exam_lamp")
    for s in range(1, 6):
        chain.apply(sample_image, severity=s, rng=np.random.default_rng(42))
        assert f"light_directional@{s}" in (chain.last_notes or "")


def test_canonical_ordering(sample_image: np.ndarray) -> None:
    """Steps are applied in canonical order (skin -> light -> blur -> noise -> jpeg) even if listed in reverse."""
    reversed_steps = [
        {"variation": "jpeg_lowres", "offset": 0, "skip_below": 1},
        {"variation": "noise_sensor", "offset": 0, "skip_below": 1},
        {"variation": "blur_motion", "offset": 0, "skip_below": 1},
        {"variation": "light_lowlight", "offset": 0, "skip_below": 1},
        {"variation": "skin_darker", "offset": 0, "skip_below": 1},
    ]
    chain = ChainVariation(name="test_canonical", steps=reversed_steps)
    canonical = chain.get_canonical_steps()
    canonical_names = [s["variation"] for s in canonical]

    expected_names = ["skin_darker", "light_lowlight", "blur_motion", "noise_sensor", "jpeg_lowres"]
    assert canonical_names == expected_names, f"Expected canonical order {expected_names}, got {canonical_names}"

    # Also verify execution notes reflect canonical order
    chain.apply(sample_image, severity=3, rng=np.random.default_rng(42))
    assert chain.last_notes == "skin_darker@3 > light_lowlight@3 > blur_motion@3 > noise_sensor@3 > jpeg_lowres@3"


def test_output_differs_from_single_steps_at_severity_5(sample_image: np.ndarray) -> None:
    """Chained output at severity 5 differs from every individual single-step output."""
    chain = get_variation("chain_dim_clinic_phone")
    chain_out = chain.apply(sample_image, severity=5, rng=np.random.default_rng(42))

    # Steps for dim_clinic_phone: light_lowlight (5), noise_sensor (5), jpeg_lowres (4)
    single_lowlight = get_variation("light_lowlight").apply(sample_image, severity=5, rng=np.random.default_rng(42))
    single_noise = get_variation("noise_sensor").apply(sample_image, severity=5, rng=np.random.default_rng(42))
    single_jpeg = get_variation("jpeg_lowres").apply(sample_image, severity=4, rng=np.random.default_rng(42))

    assert not np.array_equal(chain_out, single_lowlight), "Chain output should not match single lowlight"
    assert not np.array_equal(chain_out, single_noise), "Chain output should not match single noise"
    assert not np.array_equal(chain_out, single_jpeg), "Chain output should not match single jpeg"


def test_notes_contain_applied_steps(sample_image: np.ndarray) -> None:
    """Notes attribute contains formatted sequence of applied variations and levels."""
    chain = get_variation("chain_shaky_handheld")
    chain.apply(sample_image, severity=4, rng=np.random.default_rng(42))

    # shaky_handheld at severity 4: blur_motion@4, noise_sensor@3, jpeg_quality@3
    expected_notes = "blur_motion@4 > noise_sensor@3 > jpeg_quality@3"
    assert chain.last_notes == expected_notes, f"Expected notes '{expected_notes}', got '{chain.last_notes}'"


def test_level_1_skips_below(sample_image: np.ndarray) -> None:
    """Steps configured with skip_below are omitted at severity 1 and present at higher severity."""
    chain = get_variation("chain_dim_clinic_phone")

    # At severity 1: jpeg_lowres has skip_below: 2 and should be omitted
    chain.apply(sample_image, severity=1, rng=np.random.default_rng(42))
    assert "jpeg_lowres" not in (chain.last_notes or "")
    assert chain.last_notes == "light_lowlight@1 > noise_sensor@1"

    # At severity 2: jpeg_lowres should be included with level clip(2 - 1, 1, 5) = 1
    chain.apply(sample_image, severity=2, rng=np.random.default_rng(42))
    assert "jpeg_lowres@1" in (chain.last_notes or "")


def test_small_image_64x64_no_crash() -> None:
    """All 6 chain variations execute without error on small 64x64 input."""
    small_img = np.full((64, 64, 3), 128, dtype=np.uint8)
    chain_names = [
        "chain_dim_clinic_phone",
        "chain_shaky_handheld",
        "chain_harsh_exam_lamp",
        "chain_remote_upload",
        "chain_skin_and_light",
        "chain_random",
    ]

    for name in chain_names:
        var = get_variation(name)
        out = var.apply(small_img, severity=3, rng=np.random.default_rng(42))
        assert out.shape == (64, 64, 3), f"{name} modified image dimensions"
        assert out.dtype == np.uint8, f"{name} output dtype is not uint8"


def test_effect_direction_identical_across_levels(sample_image: np.ndarray) -> None:
    """Direction/angle choices must remain identical across levels 1-5 for the same image and seed."""
    # 1. Chain remote upload: color temperature direction stays constant
    chain_remote = get_variation("chain_remote_upload")
    remote_signs = []
    rng_remote = np.random.default_rng(42)
    for s in range(1, 6):
        chain_remote.apply(sample_image, severity=s, rng=rng_remote)
        remote_signs.append(chain_remote.last_step_details.get("light_colortemp"))
    assert len(set(remote_signs)) == 1, f"Remote upload sign varied across levels: {remote_signs}"

    # 2. Chain harsh exam lamp: directional angle stays constant
    chain_lamp = get_variation("chain_harsh_exam_lamp")
    lamp_angles = []
    rng_lamp = np.random.default_rng(42)
    for s in range(1, 6):
        chain_lamp.apply(sample_image, severity=s, rng=rng_lamp)
        lamp_angles.append(chain_lamp.last_step_details.get("light_directional"))
    assert len(set(lamp_angles)) == 1, f"Harsh exam lamp angle varied across levels: {lamp_angles}"

    # 3. Standalone light_colortemp direction stays constant
    var_col = get_variation("light_colortemp")
    col_dirs = []
    rng_col = np.random.default_rng(42)
    for s in range(1, 6):
        var_col.apply(sample_image, severity=s, rng=rng_col)
        col_dirs.append(var_col.last_direction)
    assert len(set(col_dirs)) == 1, f"Standalone light_colortemp direction varied: {col_dirs}"

    # 4. Standalone light_directional angle stays constant
    var_dir = get_variation("light_directional")
    dir_angles = []
    rng_dir = np.random.default_rng(42)
    for s in range(1, 6):
        var_dir.apply(sample_image, severity=s, rng=rng_dir)
        dir_angles.append(var_dir.last_angle_deg)
    assert len(set(dir_angles)) == 1, f"Standalone light_directional angle varied: {dir_angles}"

    # 5. Standalone blur_motion angle stays constant
    var_mot = get_variation("blur_motion")
    mot_angles = []
    rng_mot = np.random.default_rng(42)
    for s in range(1, 6):
        var_mot.apply(sample_image, severity=s, rng=rng_mot)
        mot_angles.append(var_mot.last_angle_deg)
    assert len(set(mot_angles)) == 1, f"Standalone blur_motion angle varied: {mot_angles}"

