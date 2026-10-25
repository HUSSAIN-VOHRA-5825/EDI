"""Unit tests for MediaPipe Face Mesh and facial skin mask extraction."""

from pathlib import Path
import numpy as np
import pytest
from src.face.mesh import FaceMeshDetector, get_face_skin_mask, visualize_face_mask
from src.utils.io import load_image


@pytest.fixture
def sample_image() -> np.ndarray:
    """Load or synthesize sample face image for testing."""
    sample_path = Path("data/input/sample_face.png")
    if sample_path.is_file():
        return load_image(sample_path)
    # Fallback to skimage astronaut
    import skimage.data
    return skimage.data.astronaut()


def test_face_mask_shape_and_dtype(sample_image: np.ndarray) -> None:
    """Verify facial skin mask returns correct shape (H, W), uint8 dtype, and 0/255 values."""
    mask = get_face_skin_mask(sample_image)
    assert mask is not None, "Expected face skin mask to be generated for sample face image."
    assert mask.shape == sample_image.shape[:2], "Mask dimensions must match image height and width."
    assert mask.dtype == np.uint8, "Mask dtype must be uint8."

    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255}), f"Mask must only contain binary values 0 and 255, got: {unique_vals}"
    assert np.count_nonzero(mask == 255) > 0, "Mask must contain positive skin pixels."


def test_no_face_handling_does_not_crash() -> None:
    """Verify that images with no detectable face gracefully return None without crashing."""
    blank_image = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = get_face_skin_mask(blank_image)
    assert mask is None, "Expected None for blank image with no detectable face."

    noise_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    mask_noise = get_face_skin_mask(noise_image)
    assert mask_noise is None or mask_noise.dtype == np.uint8


def test_mask_visualization(sample_image: np.ndarray, tmp_path: Path) -> None:
    """Verify mask and landmark visualization helper generates valid output."""
    detector = FaceMeshDetector()
    landmarks = detector.detect_landmarks(sample_image)
    mask = detector.generate_skin_mask(sample_image)

    out_file = tmp_path / "vis_test.png"
    vis = visualize_face_mask(
        image=sample_image,
        mask=mask,
        landmarks=landmarks,
        output_path=out_file,
    )
    assert vis.shape == sample_image.shape
    assert vis.dtype == np.uint8
    assert out_file.is_file(), "Visualization file was not written to disk."
