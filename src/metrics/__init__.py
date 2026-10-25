"""Preservation and similarity metrics module.

Evaluates clinical skin fidelity (SSIM, LPIPS), clipping fraction,
identity preservation (FaceNet cosine similarity), detail retention
(Laplacian variance ratio), and colour difference (CIE 1976 ΔE).
"""

from src.metrics.identity import FaceNetEvaluator, identity_similarity
from src.metrics.lpips_metric import LPIPSEvaluator, lpips_score
from src.metrics.preservation import check_preservation, compute_face_clipped_fraction
from src.metrics.ssim import ssim_score
from src.metrics.detail_retention import detail_retention
from src.metrics.delta_e import delta_e_face
from src.metrics.ita import compute_ita

__all__ = [
    "identity_similarity",
    "FaceNetEvaluator",
    "ssim_score",
    "lpips_score",
    "LPIPSEvaluator",
    "compute_face_clipped_fraction",
    "check_preservation",
    "detail_retention",
    "delta_e_face",
    "compute_ita",
]

