"""Reproducibility utilities for random state management.

Ensures that all stochastic operations (numpy, random, torch) yield deterministic,
reproducible results across execution runs when supplied with a seed from config.
"""

import random
from typing import Optional
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set global seeds for Python's random, NumPy, and PyTorch.

    Args:
        seed: The integer seed value loaded from configuration.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Instantiate an isolated NumPy random generator.

    Using an explicit Generator prevents variations from contaminating
    or depending on hidden global NumPy state.

    Args:
        seed: Optional integer seed for this generator.

    Returns:
        np.random.Generator instance for reproducible randomness.
    """
    return np.random.default_rng(seed)


def compute_image_identity(image: np.ndarray) -> int:
    """Compute a deterministic, resolution-invariant 32-bit hash of image content."""
    import hashlib
    sub = image[::4, ::4, :]
    return int(hashlib.md5(sub.tobytes()).hexdigest()[:8], 16)


def derive_step_seed(base_seed: int, image: np.ndarray, step_name: str) -> int:
    """Derive deterministic seed from base_seed + image identity + step name.

    Guarantees that directional attributes (angles, warm/cool signs) remain
    identical across severity levels 1–5 for the same image and seed.
    """
    import hashlib
    img_id = compute_image_identity(image)
    key = f"{int(base_seed) & 0xFFFFFFFF}_{img_id}_{step_name}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def get_direction_rng(
    image: np.ndarray,
    step_name: str,
    rng: Optional[np.random.Generator] = None,
    config_seed: Optional[int] = None,
    instance: Optional[object] = None,
) -> np.random.Generator:
    """Retrieve an isolated NumPy Generator for directional draws.

    Ensures that for the same image and initial seed, the directional draw
    (e.g., angle or warm/cool direction) is identical regardless of severity.
    """
    img_id = compute_image_identity(image)
    if config_seed is not None:
        base_seed = int(config_seed)
    elif (
        instance is not None
        and getattr(instance, "_active_image_id", None) == img_id
        and (getattr(instance, "_last_rng", None) is rng or rng is None)
        and getattr(instance, "_active_base_seed", None) is not None
    ):
        base_seed = instance._active_base_seed
    else:
        if rng is not None:
            try:
                base_seed = int(rng.bit_generator.state["state"]["state"]) & 0xFFFFFFFF
            except Exception:
                base_seed = 42
        else:
            base_seed = 42
        if instance is not None:
            instance._active_image_id = img_id
            instance._last_rng = rng
            instance._active_base_seed = base_seed

    dir_seed = derive_step_seed(base_seed, image, step_name)
    return np.random.default_rng(dir_seed)

