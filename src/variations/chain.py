"""src/variations/chain.py

Chained variations module.

Combines multiple individual degradations applied in realistic canonical sequence:
  1. skin tone -> 2. lighting -> 3. blur -> 4. noise -> 5. jpeg

Each chain operates under a single overall severity level (1 to 5).
Each component step has a configured level offset, clipped to [1, 5], and an
optional skip threshold ('skip_below: N') to ensure mild severity 1 outputs.

Registers:
  - chain_dim_clinic_phone
  - chain_shaky_handheld
  - chain_harsh_exam_lamp
  - chain_remote_upload
  - chain_skin_and_light
  - chain_random
"""

import hashlib
from typing import Any, Dict, List, Optional
import numpy as np

from src.core.base import Variation
from src.core.registry import get_variation, register_variation

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _get_subvariation_config(var_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve default configuration for sub-variations from YAML."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        try:
            from src.core.config import load_config
            cfg = load_config()
            _CONFIG_CACHE = cfg.get("variations", {})
        except Exception:
            _CONFIG_CACHE = {}
    return _CONFIG_CACHE.get(var_name)

# Canonical physical ordering: the sequence in which real-world photographic artifacts occur
FAMILY_ORDER = {
    # 0: Skin tone (intrinsic subject property)
    "skin_lighter": 0,
    "skin_darker": 0,
    "skin_warm": 0,
    "skin_cool": 0,
    # 1: Lighting (scene illumination)
    "light_lowlight": 1,
    "light_overexposed": 1,
    "light_directional": 1,
    "light_colortemp": 1,
    # 2: Optical blur (lens focus and motion)
    "blur_gaussian": 2,
    "blur_motion": 2,
    "blur_defocus": 2,
    # 3: Sensor noise (image capture noise)
    "noise_gaussian": 3,
    "noise_sensor": 3,
    # 4: Digital compression (ISP processing and transmission)
    "jpeg_quality": 4,
    "jpeg_lowres": 4,
}

# Groupings for random chain generation
FAMILY_VARIATIONS = {
    "skin_tone": ["skin_lighter", "skin_darker", "skin_warm", "skin_cool"],
    "lighting": ["light_lowlight", "light_overexposed", "light_directional", "light_colortemp"],
    "blur": ["blur_gaussian", "blur_motion", "blur_defocus"],
    "noise": ["noise_gaussian", "noise_sensor"],
    "jpeg": ["jpeg_quality", "jpeg_lowres"],
}
CANONICAL_FAMILIES = ["skin_tone", "lighting", "blur", "noise", "jpeg"]


def sort_canonical_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort a list of step dictionaries into canonical physical execution order.

    Preserves listed relative order for variations within the same family.
    """
    return sorted(steps, key=lambda s: FAMILY_ORDER.get(s.get("variation", ""), 99))


class ChainVariation(Variation):
    """Base class for executing a sequence of variations in canonical order."""

    def __init__(
        self,
        name: str,
        steps: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        cfg_steps = self.config.get("steps") if self.config else None
        self.raw_steps = cfg_steps if cfg_steps is not None else (steps or [])
        # Always order steps canonically: skin -> lighting -> blur -> noise -> jpeg
        self.canonical_steps = sort_canonical_steps(self.raw_steps)
        self.last_notes: Optional[str] = None

    def get_canonical_steps(self) -> List[Dict[str, Any]]:
        """Return steps sorted in canonical execution order."""
        return list(self.canonical_steps)

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Apply chained variation steps sequentially in canonical order.

        Args:
            image: Input RGB image (H, W, 3) uint8.
            severity: Overall severity level 1 to 5.
            mask: Optional reference face mask (passed to steps that support it).
            rng: NumPy random generator (child generators are derived deterministically).

        Returns:
            np.ndarray: Modified RGB image (H, W, 3) uint8.
        """
        from src.utils.seeding import compute_image_identity, derive_step_seed

        img_id = compute_image_identity(image)
        if hasattr(self, "_active_image_id") and self._active_image_id == img_id and (getattr(self, "_last_rng", None) is rng or rng is None):
            base_seed = self._active_base_seed
        else:
            if rng is not None:
                try:
                    base_seed = int(rng.bit_generator.state["state"]["state"]) & 0xFFFFFFFF
                except Exception:
                    base_seed = int(self.config.get("seed", 42))
            else:
                base_seed = int(self.config.get("seed", 42))
            self._active_image_id = img_id
            self._last_rng = rng
            self._active_base_seed = base_seed

        curr_img = image.copy()
        applied_notes: List[str] = []
        self.last_step_details = {}

        for step in self.canonical_steps:
            var_name = step.get("variation", "")
            offset = int(step.get("offset", 0))
            skip_below = int(step.get("skip_below", 1))

            # Skip step if severity is below threshold (keeps level 1 mild)
            if severity < skip_below:
                continue

            step_level = int(np.clip(severity + offset, 1, 5))

            # Derive step seed from base seed + image identity + step name (not from severity or shared stream)
            step_seed = derive_step_seed(base_seed, image, var_name)
            child_rng = np.random.default_rng(step_seed)

            try:
                var_cfg = self.config.get("variation_configs", {}).get(var_name)
                if var_cfg is None:
                    var_cfg = _get_subvariation_config(var_name)
                var_obj = get_variation(var_name, config=var_cfg)
                curr_img = var_obj.apply(
                    image=curr_img,
                    severity=step_level,
                    mask=mask,
                    rng=child_rng,
                )
                applied_notes.append(f"{var_name}@{step_level}")
                if hasattr(var_obj, "last_direction_str"):
                    self.last_step_details[var_name] = var_obj.last_direction_str
                elif hasattr(var_obj, "last_angle_deg"):
                    self.last_step_details[var_name] = f"{var_obj.last_angle_deg:.1f}deg"
            except Exception as e:
                # Never crash the pipeline on individual sub-step failures
                applied_notes.append(f"{var_name}@{step_level}(failed)")

        self.last_notes = " > ".join(applied_notes) if applied_notes else "none"
        return curr_img


# ──────────────────────────────────────────────────────────────────────────────
# Preset Implementations
# ──────────────────────────────────────────────────────────────────────────────

@register_variation("chain_dim_clinic_phone")
class DimClinicPhoneVariation(ChainVariation):
    """Poor indoor lighting with CMOS sensor noise and phone JPEG compression."""

    DEFAULT_STEPS = [
        {"variation": "light_lowlight", "offset": 0, "skip_below": 1},
        {"variation": "noise_sensor", "offset": 0, "skip_below": 1},
        {"variation": "jpeg_lowres", "offset": -1, "skip_below": 2},
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="chain_dim_clinic_phone", steps=self.DEFAULT_STEPS, config=config)


@register_variation("chain_shaky_handheld")
class ShakyHandheldVariation(ChainVariation):
    """Handheld camera motion blur with sensor noise and lossy compression."""

    DEFAULT_STEPS = [
        {"variation": "blur_motion", "offset": 0, "skip_below": 1},
        {"variation": "noise_sensor", "offset": -1, "skip_below": 2},
        {"variation": "jpeg_quality", "offset": -1, "skip_below": 2},
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="chain_shaky_handheld", steps=self.DEFAULT_STEPS, config=config)


@register_variation("chain_harsh_exam_lamp")
class HarshExamLampVariation(ChainVariation):
    """Directional lamp casting harsh shadows and overexposed hot spots."""

    DEFAULT_STEPS = [
        {"variation": "light_directional", "offset": 0, "skip_below": 1},
        {"variation": "light_overexposed", "offset": -1, "skip_below": 2},
        {"variation": "jpeg_lowres", "offset": -2, "skip_below": 3},
    ]

    # Boosted directional shadow for this chain so the left-right brightness difference
    # remains clearly visible at levels 4-5 despite subsequent overexposure washout
    CUSTOM_VARIATION_CONFIGS = {
        "light_directional": {
            "levels": {
                1: {"shadow_strength": 0.35},
                2: {"shadow_strength": 0.55},
                3: {"shadow_strength": 0.75},
                4: {"shadow_strength": 0.90},
                5: {"shadow_strength": 0.98},
            }
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        var_cfgs = dict(cfg.get("variation_configs", {}))
        for k, v in self.CUSTOM_VARIATION_CONFIGS.items():
            var_cfgs.setdefault(k, v)
        cfg["variation_configs"] = var_cfgs
        super().__init__(name="chain_harsh_exam_lamp", steps=self.DEFAULT_STEPS, config=cfg)


@register_variation("chain_remote_upload")
class RemoteUploadVariation(ChainVariation):
    """Color temperature cast with defocus blur and low-resolution upload compression."""

    DEFAULT_STEPS = [
        {"variation": "light_colortemp", "offset": 0, "skip_below": 1},
        {"variation": "blur_defocus", "offset": -2, "skip_below": 3},
        {"variation": "jpeg_lowres", "offset": -2, "skip_below": 3},
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="chain_remote_upload", steps=self.DEFAULT_STEPS, config=config)


@register_variation("chain_skin_and_light")
class SkinAndLightVariation(ChainVariation):
    """Skin tone darkening in clinical low-light with CMOS read noise."""

    DEFAULT_STEPS = [
        {"variation": "skin_darker", "offset": 0, "skip_below": 1},
        {"variation": "light_lowlight", "offset": -1, "skip_below": 2},
        {"variation": "noise_sensor", "offset": -1, "skip_below": 2},
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="chain_skin_and_light", steps=self.DEFAULT_STEPS, config=config)


def _compute_image_identity(image: np.ndarray) -> int:
    """Compute deterministic hash of image content."""
    sub = image[::4, ::4, :]
    return int(hashlib.md5(sub.tobytes()).hexdigest()[:8], 16)


@register_variation("chain_random")
class RandomChainVariation(Variation):
    """Randomly samples 2 to 3 distinct degradation families and applies them in canonical order."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, base_seed: int = 42) -> None:
        super().__init__(name="chain_random", config=config)
        self.base_seed = int(self.config.get("seed", base_seed))
        self.last_notes: Optional[str] = None
        self._current_recipe: Optional[List[Dict[str, Any]]] = None
        self._current_image_id: Optional[int] = None
        self._last_rng: Optional[np.random.Generator] = None

    def get_recipe(self, image: np.ndarray, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Choose variation recipe deterministically once per image and seed."""
        img_id = _compute_image_identity(image)
        eff_seed = seed if seed is not None else self.base_seed
        recipe_seed = ((int(eff_seed) & 0xFFFFFFFF) * 1000003 + img_id) & 0xFFFFFFFF
        recipe_rng = np.random.default_rng(recipe_seed)

        k = int(recipe_rng.choice([2, 3]))
        chosen_indices = sorted(recipe_rng.choice(len(CANONICAL_FAMILIES), size=k, replace=False))
        chosen_families = [CANONICAL_FAMILIES[i] for i in chosen_indices]

        steps = []
        for fam in chosen_families:
            var_candidates = FAMILY_VARIATIONS[fam]
            chosen_var = str(recipe_rng.choice(var_candidates))
            steps.append({"variation": chosen_var, "offset": 0, "skip_below": 1})
        return steps

    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(self.base_seed)

        img_id = _compute_image_identity(image)
        # Select recipe once per image and seed:
        if seed is not None:
            steps = self.get_recipe(image, seed=seed)
        elif self._current_image_id == img_id and self._current_recipe is not None and (rng is self._last_rng or rng is None):
            steps = self._current_recipe
        else:
            eff_seed = self.base_seed
            if rng is not None:
                try:
                    eff_seed = int(rng.bit_generator.state["state"]["state"]) & 0xFFFFFFFF
                except Exception:
                    eff_seed = self.base_seed
            steps = self.get_recipe(image, seed=eff_seed)
            self._current_image_id = img_id
            self._current_recipe = steps
            self._last_rng = rng

        executor = ChainVariation(name="chain_random_exec", steps=steps)
        result = executor.apply(image=image, severity=severity, mask=mask, rng=rng)
        self.last_notes = executor.last_notes
        return result
