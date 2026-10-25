"""Facial identity similarity evaluation using FaceNet embeddings.

In medical AI stress-testing, variations (e.g., lighting, blur, skin tone shifts) must not
alter the patient's underlying biometric identity. We crop and align faces using MTCNN,
pass them through an InceptionResnetV1 model pre-trained on VGGFace2, and compute
the cosine similarity between the 512-dimensional feature embeddings.
"""

import logging
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

logger = logging.getLogger(__name__)


class FaceNetEvaluator:
    """Manages FaceNet models (MTCNN face detector + InceptionResnetV1).

    Models are loaded into memory once and reused across all pipeline iterations
    to prevent repetitive disk IO and GPU/CPU initialization overhead.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """Initialize MTCNN and FaceNet InceptionResnetV1 models.

        Args:
            device: 'cuda', 'cpu', or None (auto-selects best available device).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # MTCNN crops and standardizes faces to 160x160 pixels
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            device=self.device,
            post_process=True,
        )

        # Pre-trained VGGFace2 feature extractor
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def extract_embedding(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """Extract a 512-dim identity embedding vector from an image.

        Args:
            image: RGB image as uint8 NumPy array (H, W, 3).

        Returns:
            torch.Tensor of shape (1, 512) or None if no face is detected by MTCNN.
        """
        # MTCNN accepts uint8 NumPy array in RGB format
        face_tensor = self.mtcnn(image)
        if face_tensor is None:
            return None

        # Add batch dimension and transfer to device
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.resnet(face_tensor)
            # Normalize embedding vector to unit length for cosine similarity
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def compute_similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        """Calculate cosine similarity of identity embeddings between two images.

        Args:
            img_a: First RGB image (H, W, 3).
            img_b: Second RGB image (H, W, 3).

        Returns:
            float: Cosine similarity in range [-1.0, 1.0]. Returns 0.0 if either image
            does not contain a detectable face.
        """
        emb_a = self.extract_embedding(img_a)
        if emb_a is None:
            logger.warning("Identity check: No face detected in reference image (img_a).")
            return 0.0

        emb_b = self.extract_embedding(img_b)
        if emb_b is None:
            logger.warning("Identity check: No face detected in compared image (img_b).")
            return 0.0

        # Cosine similarity between unit-normalized vectors is simply their dot product
        similarity = float(torch.sum(emb_a * emb_b).clamp(-1.0, 1.0).item())
        return similarity


# Singleton instance loaded once and reused across calls
_FACENET_EVALUATOR: Optional[FaceNetEvaluator] = None


def get_default_facenet_evaluator() -> FaceNetEvaluator:
    """Retrieve or lazily initialize the singleton FaceNetEvaluator instance."""
    global _FACENET_EVALUATOR
    if _FACENET_EVALUATOR is None:
        _FACENET_EVALUATOR = FaceNetEvaluator()
    return _FACENET_EVALUATOR


def identity_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute cosine similarity of FaceNet embeddings between two face images.

    Args:
        img_a: RGB image as uint8 array (H, W, 3).
        img_b: RGB image as uint8 array (H, W, 3).

    Returns:
        float: Cosine similarity score [-1.0, 1.0], or 0.0 if face detection fails.
    """
    evaluator = get_default_facenet_evaluator()
    return evaluator.compute_similarity(img_a, img_b)
