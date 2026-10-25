"""Individual Typology Angle (ITA) metric for dermatological skin tone assessment.

ITA = arctan((L* - 50) / b*) * 180 / pi
Calculated from the median L* and b* inside the facial skin mask.
Standard dermatological classification:
  > 55°: Very light
  41° to 55°: Light
  28° to 41°: Intermediate
  10° to 28°: Tan
  -30° to 10°: Brown
  < -30°: Dark
"""

from typing import Optional
import numpy as np
from skimage.color import rgb2lab


def compute_ita(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Individual Typology Angle (ITA) in degrees from median L* and b* inside mask.

    Args:
        image: RGB image as uint8 NumPy array (H, W, 3).
        mask: Optional binary mask (H, W). If provided, calculates over mask > 0.

    Returns:
        float: ITA angle in degrees, rounded to 2 decimal places.
    """
    if mask is not None and np.count_nonzero(mask > 0) > 0:
        pixels = image[mask > 0]
    else:
        pixels = image.reshape(-1, 3)

    if pixels.size == 0:
        return 0.0

    # Convert RGB [0, 255] to CIELAB float [L*: 0..100, b*: -128..127]
    lab = rgb2lab(pixels.reshape(1, -1, 3).astype(np.float32) / 255.0)[0]
    median_L = float(np.median(lab[:, 0]))
    median_b = float(np.median(lab[:, 2]))

    # Avoid divide-by-zero if b* is nearly 0
    if abs(median_b) < 1e-5:
        median_b = 1e-5 if median_b >= 0 else -1e-5

    ita_rad = np.arctan((median_L - 50.0) / median_b)
    ita_deg = float(ita_rad * 180.0 / np.pi)
    return round(ita_deg, 2)
