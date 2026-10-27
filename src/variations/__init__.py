"""Variations package.

Exposes and registers all concrete Variation implementations.
"""

from src.variations.blur_defocus import DefocusBlurVariation
from src.variations.blur_gaussian import GaussianBlurVariation
from src.variations.blur_motion import MotionBlurVariation
from src.variations.dummy import BrightnessTestVariation, IdentityTestVariation
from src.variations.jpeg_lowres import JpegLowResVariation
from src.variations.jpeg_quality import JpegQualityVariation
from src.variations.noise_gaussian import GaussianNoiseVariation
from src.variations.noise_sensor import SensorNoiseVariation
from src.variations.lighting import (
    LowLightVariation,
    OverexposedVariation,
    DirectionalLightVariation,
    ColourTemperatureVariation,
)
from src.variations.skin_tone import (
    SkinLighterVariation,
    SkinDarkerVariation,
    SkinWarmVariation,
    SkinCoolVariation,
)
from src.variations.chain import (
    ChainVariation,
    DimClinicPhoneVariation,
    ShakyHandheldVariation,
    HarshExamLampVariation,
    RemoteUploadVariation,
    SkinAndLightVariation,
    RandomChainVariation,
)
from src.variations.aging import AgingVariation

__all__ = [
    "AgingVariation",
    "IdentityTestVariation",
    "BrightnessTestVariation",
    "GaussianBlurVariation",
    "MotionBlurVariation",
    "DefocusBlurVariation",
    "GaussianNoiseVariation",
    "SensorNoiseVariation",
    "JpegQualityVariation",
    "JpegLowResVariation",
    "LowLightVariation",
    "OverexposedVariation",
    "DirectionalLightVariation",
    "ColourTemperatureVariation",
    "SkinLighterVariation",
    "SkinDarkerVariation",
    "SkinWarmVariation",
    "SkinCoolVariation",
    "ChainVariation",
    "DimClinicPhoneVariation",
    "ShakyHandheldVariation",
    "HarshExamLampVariation",
    "RemoteUploadVariation",
    "SkinAndLightVariation",
    "RandomChainVariation",
]
