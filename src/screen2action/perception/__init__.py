"""Typed real, cached, and oracle perception interfaces."""

from screen2action.perception.base import TextRecognizer, TextRegionDetector, UiDetector
from screen2action.perception.cache import ContentAddressedPerceptionCache, PerceptionCacheKey
from screen2action.perception.loader import CachedPerceptionLoader
from screen2action.perception.oracle import OracleFrame, OraclePerception
from screen2action.perception.pipeline import (
    FullScreenPerception,
    FullScreenPerceptionConfig,
    PerceptionFrame,
)
from screen2action.perception.precompute import (
    PerceptionBundleSpec,
    precompute_perception,
    validate_perception_cache,
)

__all__ = [
    "ContentAddressedPerceptionCache",
    "CachedPerceptionLoader",
    "FullScreenPerception",
    "FullScreenPerceptionConfig",
    "OracleFrame",
    "OraclePerception",
    "PerceptionCacheKey",
    "PerceptionBundleSpec",
    "PerceptionFrame",
    "TextRecognizer",
    "TextRegionDetector",
    "UiDetector",
    "precompute_perception",
    "validate_perception_cache",
]
