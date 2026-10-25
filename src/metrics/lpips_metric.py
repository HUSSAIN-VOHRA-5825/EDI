"""Learned Perceptual Image Patch Similarity (LPIPS) metric.

LPIPS calculates perceptual distance by measuring deep feature differences in
a pre-trained network (AlexNet). Unlike shallow pixel metrics (MSE, PSNR), LPIPS
mirrors human visual perception of distortions, textures, and artifacts.
"""

import logging
from typing import Optional
import cv2
import lpips
import numpy as np
import torch

logger = logging.getLogger(__name__)


class LPIPSEvaluator:
    """Manages pre-trained LPIPS model loaded once and reused across evaluations."""

    def __init__(self, net: str = "alex", device: Optional[str] = None) -> None:
        """Initialize LPIPS network in evaluation mode.

        Args:
            net: Backbone network architecture ('alex' is standard for speed and accuracy).
            device: 'cuda', 'cpu', or None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize network and set to eval mode
        self.loss_fn = lpips.LPIPS(net=net, verbose=False).eval().to(self.device)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert uint8 RGB NumPy array (H, W, 3) to normalized PyTorch tensor (1, 3, H, W).

        LPIPS expects tensor values in range [-1.0, 1.0].
        """
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1.0
        return tensor.unsqueeze(0).to(self.device)

    def compute_distance(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate LPIPS perceptual distance between two images.

        Args:
            img_a: Reference RGB image as uint8 NumPy array (H, W, 3).
            img_b: Compared RGB image as uint8 NumPy array (H, W, 3).
            mask: Optional binary uint8 mask (H, W). If provided, crops to the mask's
                bounding box and isolates the masked skin region for comparison.

        Returns:
            float: Perceptual distance (0.0 represents identical images).
        """
        # Ensure dimensions match
        if img_a.shape[:2] != img_b.shape[:2]:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Case 1: Masked region of interest
        if mask is not None:
            if mask.shape[:2] != img_a.shape[:2]:
                mask = cv2.resize(mask, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_NEAREST)

            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0:
                logger.warning("LPIPS: Mask has 0 positive pixels. Evaluating full image.")
                tensor_a = self._to_tensor(img_a)
                tensor_b = self._to_tensor(img_b)
            else:
                # Crop to bounding box of the masked region
                ymin, ymax = y_indices.min(), y_indices.max() + 1
                xmin, xmax = x_indices.min(), x_indices.max() + 1

                # Zero out background outside the mask so only skin features contribute
                patch_a = img_a[ymin:ymax, xmin:xmax].copy()
                patch_b = img_b[ymin:ymax, xmin:xmax].copy()
                sub_mask = (mask[ymin:ymax, xmin:xmax] > 0)[:, :, None]

                patch_a[~sub_mask[:, :, 0]] = 0
                patch_b[~sub_mask[:, :, 0]] = 0

                tensor_a = self._to_tensor(patch_a)
                tensor_b = self._to_tensor(patch_b)
        else:
            tensor_a = self._to_tensor(img_a)
            tensor_b = self._to_tensor(img_b)

        with torch.no_grad():
            distance = self.loss_fn(tensor_a, tensor_b)
            return float(distance.item())


# Singleton instance loaded once and reused across evaluations
_LPIPS_EVALUATOR: Optional[LPIPSEvaluator] = None


def get_default_lpips_evaluator() -> LPIPSEvaluator:
    """Retrieve or lazily initialize the singleton LPIPSEvaluator instance."""
    global _LPIPS_EVALUATOR
    if _LPIPS_EVALUATOR is None:
        _LPIPS_EVALUATOR = LPIPSEvaluator()
    return _LPIPS_EVALUATOR


def lpips_score(
    img_a: np.ndarray,
    img_b: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute LPIPS perceptual distance between two images.

    Args:
        img_a: RGB image as uint8 array (H, W, 3).
        img_b: RGB image as uint8 array (H, W, 3).
        mask: Optional binary uint8 mask (H, W).

    Returns:
        float: LPIPS distance >= 0.0 (0.0 indicates perceptually identical).
    """
    evaluator = get_default_lpips_evaluator()
    return evaluator.compute_distance(img_a, img_b, mask=mask)
