"""src/evaluation/models.py

Common interface for the three models under test in the robustness evaluation:
  1. FaceDetectionModel  — MediaPipe Face Mesh; metrics: detection rate, NME
  2. IdentityModel        — facenet-pytorch; metric: same-person match rate
  3. ClassifierModel      — ResNet18 fine-tuned with 5-fold stratified CV; metrics: accuracy, F1, consistency

All models expose:
    model.fit(images, labels, filenames)   → (ClassifierModel only)
    model.predict(images, originals, ...)  → list[dict]  one dict per image

"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from PIL import Image
import cv2
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_pil(img: np.ndarray) -> Image.Image:
    """Convert uint8 RGB ndarray to PIL Image."""
    return Image.fromarray(img)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Face Detection (MediaPipe Face Mesh)
# ──────────────────────────────────────────────────────────────────────────────

class FaceDetectionModel:
    """MediaPipe Face Mesh — detection success rate and normalised landmark error (NME).

    NME = mean Euclidean distance between landmarks found on the varied image and
    those found on the corresponding original image, normalised by the inter-ocular
    distance (IOD) of the original.  If detection fails on the varied image, NME is
    set to NaN and the image counts as a detection failure.
    """

    # Left/right eye centre landmark indices (from MediaPipe 468-point mesh)
    _LEFT_EYE_IDX = 33   # inner corner of left eye
    _RIGHT_EYE_IDX = 263  # inner corner of right eye

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self.min_detection_confidence = min_detection_confidence
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    def _landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Return (N, 2) pixel landmarks or None."""
        h, w = image.shape[:2]
        res = self._mesh.process(image)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)

    def _iod(self, landmarks: np.ndarray) -> float:
        """Inter-ocular distance from landmark array."""
        return float(np.linalg.norm(
            landmarks[self._LEFT_EYE_IDX] - landmarks[self._RIGHT_EYE_IDX]
        ))

    def predict(
        self,
        images: Sequence[np.ndarray],
        orig_landmarks: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return per-image metrics.

        Args:
            images: List of RGB uint8 images.
            orig_landmarks: Pre-computed landmarks for the originals (avoids re-detection).

        Returns:
            List of dicts with keys: detected (bool), nme (float or nan).
        """
        results = []
        for i, img in enumerate(images):
            lm = self._landmarks(img)
            detected = lm is not None
            nme = float("nan")
            if detected and orig_landmarks is not None and orig_landmarks[i] is not None:
                orig_lm = orig_landmarks[i]
                iod = self._iod(orig_lm)
                if iod > 0:
                    # Only compare the 468 common landmarks (refine adds iris points at 468+)
                    n = min(len(lm), len(orig_lm), 468)
                    nme = float(np.mean(np.linalg.norm(lm[:n] - orig_lm[:n], axis=1)) / iod)
            results.append({"detected": detected, "nme": nme})
        return results

    def get_landmarks(self, images: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Cache-friendly bulk landmark extraction."""
        return [self._landmarks(img) for img in images]

    def close(self) -> None:
        self._mesh.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Identity Model (facenet-pytorch)
# ──────────────────────────────────────────────────────────────────────────────

class IdentityModel:
    """FaceNet-based identity verification.

    Metric: same-person match rate — fraction of varied images whose cosine
    similarity to the original embedding is >= identity_similarity_min.
    """

    _TRANSFORM = T.Compose([
        T.Resize((160, 160)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def __init__(self, identity_similarity_min: float = 0.60) -> None:
        self.threshold = identity_similarity_min
        self.model = InceptionResnetV1(pretrained="vggface2").eval()

    @torch.no_grad()
    def embed(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return (N, D) embedding matrix."""
        tensors = torch.stack([self._TRANSFORM(_to_pil(img)) for img in images])
        return self.model(tensors).numpy()

    def predict(
        self,
        images: Sequence[np.ndarray],
        orig_embeddings: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Return per-image identity similarity and match flag.

        Args:
            images: Varied images (N,).
            orig_embeddings: (N, D) embeddings of the corresponding originals.
        """
        varied_embs = self.embed(images)
        results = []
        for i, emb in enumerate(varied_embs):
            sim = _cosine_sim(emb, orig_embeddings[i])
            results.append({"identity_similarity": sim, "matched": sim >= self.threshold})
        return results

    def get_embeddings(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Bulk embedding extraction for caching."""
        return self.embed(images)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Classifier (ResNet18, 5-fold stratified CV)
# ──────────────────────────────────────────────────────────────────────────────

class ClassifierModel:
    """ImageNet-pretrained ResNet18 fine-tuned for binary age classification.

    Training: only the final FC layer (and optionally the last ResNet block) are
    fine-tuned on the ORIGINAL images from the training folds.  Variations are
    NEVER shown during training — they are only evaluated on the held-out fold.

    Labels: 'under_30' → 0, '30_plus' → 1.
    """

    LABEL_MAP = {"under_30": 0, "30_plus": 1}
    INV_LABEL_MAP = {0: "under_30", 1: "30_plus"}

    _TRAIN_TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _EVAL_TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, epochs: int = 5, lr: float = 1e-3, seed: int = 42, fine_tune_block: bool = True) -> None:
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.fine_tune_block = fine_tune_block
        self._model: Optional[nn.Module] = None

    def _build_model(self) -> nn.Module:
        import torchvision.models as models
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Freeze all layers
        for p in net.parameters():
            p.requires_grad = False
        # Unfreeze last residual block (layer4) and final FC
        if self.fine_tune_block:
            for p in net.layer4.parameters():
                p.requires_grad = True
        net.fc = nn.Linear(net.fc.in_features, 2)  # Binary classification head
        return net

    def fit(
        self,
        images: Sequence[np.ndarray],
        labels: Sequence[str],
        batch_size: int = 16,
    ) -> None:
        """Fine-tune on original-image training fold.

        Args:
            images: Training images (originals only — no variations).
            labels: String labels ('under_30' or '30_plus').
        """
        torch.manual_seed(self.seed)
        net = self._build_model()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        tensors = torch.stack([self._TRAIN_TRANSFORM(_to_pil(img)) for img in images])
        y = torch.tensor([self.LABEL_MAP[l] for l in labels], dtype=torch.long)
        dataset = TensorDataset(tensors, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()

        net.eval()
        self._model = net

    @torch.no_grad()
    def predict_proba(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return (N, 2) softmax probabilities."""
        assert self._model is not None, "Call fit() before predict_proba()"
        tensors = torch.stack([self._EVAL_TRANSFORM(_to_pil(img)) for img in images])
        logits = self._model(tensors)
        return torch.softmax(logits, dim=1).numpy()

    def predict(
        self,
        images: Sequence[np.ndarray],
        orig_predictions: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Return per-image predictions and consistency flags.

        Args:
            images: Images to classify (may include variations).
            orig_predictions: Model predictions on the corresponding original images
                              (int class indices).  If provided, consistency is computed.
        """
        proba = self.predict_proba(images)
        preds = proba.argmax(axis=1)
        results = []
        for i, (pred, prob) in enumerate(zip(preds, proba)):
            rec: Dict[str, Any] = {
                "prediction": int(pred),
                "label_str": self.INV_LABEL_MAP[int(pred)],
                "confidence": float(prob.max()),
            }
            if orig_predictions is not None:
                rec["consistent"] = int(pred) == orig_predictions[i]
            results.append(rec)
        return results
