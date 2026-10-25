"""Unit tests for preservation and similarity metrics (Identity, SSIM, LPIPS)."""

from pathlib import Path
import numpy as np
import pytest
from src.metrics.identity import identity_similarity
from src.metrics.lpips_metric import lpips_score
from src.metrics.preservation import check_preservation
from src.metrics.ssim import ssim_score
from src.utils.io import load_image


@pytest.fixture
def sample_image() -> np.ndarray:
    """Load sample face image for testing."""
    sample_path = Path("data/input/sample_face.png")
    if sample_path.is_file():
        return load_image(sample_path)
    import skimage.data
    return skimage.data.astronaut()


def test_identity_similarity_self(sample_image: np.ndarray) -> None:
    """Identity similarity of an image compared with itself should be ~1.0."""
    sim = identity_similarity(sample_image, sample_image)
    assert isinstance(sim, float)
    assert sim >= 0.98, f"Expected self identity similarity close to 1.0, got: {sim}"


def test_identity_similarity_no_face_handling() -> None:
    """Blank or non-face images should return 0.0 without crashing."""
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    sim = identity_similarity(blank, blank)
    assert sim == 0.0, f"Expected 0.0 for non-face image, got: {sim}"


def test_ssim_score_self(sample_image: np.ndarray) -> None:
    """SSIM of an image compared with itself should be exactly 1.0."""
    h, w = sample_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255

    score_full = ssim_score(sample_image, sample_image, mask=None)
    score_masked = ssim_score(sample_image, sample_image, mask=mask)

    assert pytest.approx(score_full, rel=1e-4) == 1.0
    assert pytest.approx(score_masked, rel=1e-4) == 1.0


def test_lpips_score_self(sample_image: np.ndarray) -> None:
    """LPIPS perceptual distance of an image compared with itself should be ~0.0."""
    h, w = sample_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255

    dist_full = lpips_score(sample_image, sample_image, mask=None)
    dist_masked = lpips_score(sample_image, sample_image, mask=mask)

    assert dist_full < 0.02, f"Expected LPIPS close to 0.0, got: {dist_full}"
    assert dist_masked < 0.02, f"Expected masked LPIPS close to 0.0, got: {dist_masked}"


def test_check_preservation_self(sample_image: np.ndarray) -> None:
    """check_preservation on identical image must pass all metrics and overall."""
    h, w = sample_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255

    res = check_preservation(sample_image, sample_image, mask=mask)
    assert res["passed"] is True
    assert res["passes"]["identity"] is True
    assert res["passes"]["ssim_face"] is True
    assert res["passes"]["ssim_full"] is True
    assert res["passes"]["lpips_face"] is True
    assert res["identity_similarity"] >= 0.98
    assert res["ssim_face"] >= 0.99
    assert res["lpips_face"] <= 0.02
    assert "face_clipped_fraction" in res
    assert 0.0 <= res["face_clipped_fraction"] <= 1.0


def test_compute_face_clipped_fraction() -> None:
    """Verify face_clipped_fraction accurately measures saturated/crushed pixel ratios."""
    from src.metrics.preservation import compute_face_clipped_fraction

    # 10x10 image with 100 pixels
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    # Clip 10 pixels to 255 (10% clipped)
    img[:1, :, 0] = 255

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, :] = 255  # all pixels inside mask

    frac = compute_face_clipped_fraction(img, mask=mask)
    assert pytest.approx(frac, abs=1e-4) == 0.10

    # Clip 5 pixels to 0 in another channel
    img[1, :5, 1] = 0
    frac2 = compute_face_clipped_fraction(img, mask=mask)
    assert pytest.approx(frac2, abs=1e-4) == 0.15
