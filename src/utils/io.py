"""Image input/output and geometric preprocessing utilities."""

from pathlib import Path
from typing import List, Union
import cv2
import numpy as np
from PIL import Image


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk as an RGB uint8 NumPy array.

    Args:
        path: Path to the image file.

    Returns:
        np.ndarray: Image array with shape (H, W, 3) in RGB order and dtype uint8.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at path: {path}")

    # Use PIL to read image reliably across OS platforms and handle color profiles
    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        return np.array(img_rgb, dtype=np.uint8)


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """Save an RGB NumPy array to disk as an image file.

    Creates parent directories automatically if they do not exist.

    Args:
        image: Image array with shape (H, W, 3) in RGB order and dtype uint8.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_pil = Image.fromarray(image)
    img_pil.save(path)


def resize_keep_aspect_ratio(
    image: np.ndarray,
    target_longer_side: int = 512,
    interpolation: str = "lanczos"
) -> np.ndarray:
    """Resize image so its longer side matches target_longer_side, preserving aspect ratio.

    If the image is already smaller or equal on both sides, we still resize to
    standardize input dimensions unless it exactly matches target_longer_side.

    Args:
        image: Input RGB image (H, W, 3).
        target_longer_side: Pixel length of the longer dimension.
        interpolation: Resampling method ('lanczos', 'bilinear', 'nearest', 'bicubic').

    Returns:
        np.ndarray: Rescaled image.
    """
    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side == target_longer_side:
        return image.copy()

    scale = target_longer_side / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp_map = {
        "lanczos": cv2.INTER_LANCZOS4,
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }
    cv_interp = interp_map.get(interpolation.lower(), cv2.INTER_LANCZOS4)
    return cv2.resize(image, (new_w, new_h), interpolation=cv_interp)


def get_image_paths(directory: Union[str, Path]) -> List[Path]:
    """Find and return all valid image file paths in a directory.

    Supported extensions: .jpg, .jpeg, .png, .bmp, .webp, .tiff.

    Args:
        directory: Directory to search.

    Returns:
        List of Path objects sorted alphabetically.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    image_paths = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    return sorted(image_paths)
