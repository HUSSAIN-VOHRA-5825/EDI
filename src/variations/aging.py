"""FRAN Face Re-aging Variation implementation."""

from typing import Any, Dict, Optional
import numpy as np
from src.core.base import Variation
from src.core.registry import register_variation
from src.face.aging import reage_face


@register_variation("facial_aging")
class AgingVariation(Variation):
    """Controlled facial aging variation using FRAN U-Net ONNX model.

    Severity levels map to progressive age increments:
        Level 1: +10 years
        Level 2: +20 years
        Level 3: +30 years
        Level 4: +40 years
        Level 5: +50 years
    """

    def __init__(self, name: str = "facial_aging", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, config=config)

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        source_age: int = 25,
        target_age: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply facial aging to the input image.

        Args:
            image: Input RGB image as uint8 NumPy array (H, W, 3).
            severity: Severity level (1 to 5).
            mask: Optional facial mask.
            rng: Optional NumPy random generator.
            source_age: Subject's estimated current age (default 25).
            target_age: Optional explicit target age. If None, derived from severity level.

        Returns:
            Aged RGB image as uint8 NumPy array (H, W, 3).
        """
        params = self.get_level_params(severity)
        progression = params.get("age_progression", severity * 10)

        if target_age is None:
            target_age = min(95, source_age + progression)

        return reage_face(
            image=image,
            source_age=source_age,
            target_age=target_age,
        )
