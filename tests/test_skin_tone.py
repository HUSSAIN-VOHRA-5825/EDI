"""Unit tests for skin tone variations (skin_lighter, skin_darker, skin_warm, skin_cool)."""

from pathlib import Path
import cv2
import numpy as np
import pytest
from skimage.color import rgb2lab

from src.core.registry import get_variation
from src.face.mesh import get_face_skin_mask, get_skin_edit_mask
from src.metrics.detail_retention import detail_retention
from src.metrics.ita import compute_ita
from src.utils.io import load_image


@pytest.fixture
def sample_face() -> np.ndarray:
    """Load reference face image."""
    for p in ["data/input/person_1.jpeg", "data/input/sample_face.png"]:
        path = Path(p)
        if path.is_file():
            img = load_image(path)
            # Resize to reasonable resolution for quick test execution
            h, w = img.shape[:2]
            if max(h, w) > 512:
                scale = 512.0 / max(h, w)
                img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
            return img
    # Fallback to skimage astronaut
    import skimage.data
    return skimage.data.astronaut()


def test_skin_tone_shape_and_dtype(sample_face: np.ndarray) -> None:
    """Verify output retains the exact shape and uint8 dtype of the input."""
    for var_name in ["skin_lighter", "skin_darker", "skin_warm", "skin_cool"]:
        var = get_variation(var_name)
        out = var.apply(sample_face, severity=3)
        assert out.shape == sample_face.shape, f"{var_name} altered image shape."
        assert out.dtype == np.uint8, f"{var_name} output dtype must be uint8."


def test_skin_tone_deterministic(sample_face: np.ndarray) -> None:
    """Verify skin tone operations are strictly deterministic."""
    var = get_variation("skin_lighter")
    res1 = var.apply(sample_face, severity=2)
    res2 = var.apply(sample_face, severity=2)
    np.testing.assert_array_equal(res1, res2, err_msg="skin_lighter is not deterministic across runs.")


def test_pixels_outside_dilated_mask_unchanged(sample_face: np.ndarray) -> None:
    """Verify that pixels outside the dilated edit mask remain completely unchanged."""
    edit_mask = get_skin_edit_mask(sample_face)
    assert edit_mask is not None and np.count_nonzero(edit_mask) > 0

    h, w = sample_face.shape[:2]
    sigma = max(1.0, 0.005 * w)
    ksize = int(round(sigma * 4)) * 2 + 1
    dilated_edit = cv2.dilate(edit_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)))
    outside_mask = dilated_edit == 0

    var = get_variation("skin_darker")
    modified = var.apply(sample_face, severity=5)

    max_diff = np.max(np.abs(modified[outside_mask].astype(np.int32) - sample_face[outside_mask].astype(np.int32)))
    assert max_diff == 0, f"Pixels outside dilated edit mask were altered (max diff: {max_diff})."


def test_ita_monotonicity(sample_face: np.ndarray) -> None:
    """Verify ITA strictly increases for skin_lighter and strictly decreases for skin_darker."""
    mask = get_face_skin_mask(sample_face)
    assert mask is not None and np.count_nonzero(mask) > 0

    lighter = get_variation("skin_lighter")
    darker = get_variation("skin_darker")

    ita_lighter = [compute_ita(sample_face, mask)]
    for lvl in range(1, 6):
        out = lighter.apply(sample_face, severity=lvl)
        ita_lighter.append(compute_ita(out, mask))

    for i in range(len(ita_lighter) - 1):
        assert ita_lighter[i + 1] > ita_lighter[i], (
            f"ITA failed to strictly increase for skin_lighter: {ita_lighter}"
        )

    ita_darker = [compute_ita(sample_face, mask)]
    for lvl in range(1, 6):
        out = darker.apply(sample_face, severity=lvl)
        ita_darker.append(compute_ita(out, mask))

    for i in range(len(ita_darker) - 1):
        assert ita_darker[i + 1] < ita_darker[i], (
            f"ITA failed to strictly decrease for skin_darker: {ita_darker}"
        )


def test_detail_retention_preserved(sample_face: np.ndarray) -> None:
    """Verify detail_retention is at least 0.9 at every level across all skin tone variations."""
    mask = get_face_skin_mask(sample_face)
    assert mask is not None and np.count_nonzero(mask) > 0

    for var_name in ["skin_lighter", "skin_darker", "skin_warm", "skin_cool"]:
        var = get_variation(var_name)
        for lvl in range(1, 6):
            out = var.apply(sample_face, severity=lvl)
            dr = detail_retention(sample_face, out, mask=mask)
            assert dr >= 0.90, f"{var_name} at level {lvl} dropped detail retention below 0.90 (got {dr:.4f})."


def test_runs_on_small_image_without_crashing() -> None:
    """Verify that 64x64 images without detectable faces return unchanged without crashing."""
    small_img = np.zeros((64, 64, 3), dtype=np.uint8)
    for var_name in ["skin_lighter", "skin_darker", "skin_warm", "skin_cool"]:
        var = get_variation(var_name)
        out = var.apply(small_img, severity=1)
        assert out.shape == (64, 64, 3)
        np.testing.assert_array_equal(out, small_img)
