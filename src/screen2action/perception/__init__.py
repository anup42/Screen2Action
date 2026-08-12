"""Typed real, cached, and oracle perception interfaces."""

from screen2action.perception.base import TextRecognizer, TextRegionDetector, UiDetector
from screen2action.perception.cache import ContentAddressedPerceptionCache, PerceptionCacheKey
from screen2action.perception.oracle import OracleFrame, OraclePerception
from screen2action.perception.pipeline import (
    FullScreenPerception,
    FullScreenPerceptionConfig,
    PerceptionFrame,
)

__all__ = [
    "ContentAddressedPerceptionCache",
    "FullScreenPerception",
    "FullScreenPerceptionConfig",
    "OracleFrame",
    "OraclePerception",
    "PerceptionCacheKey",
    "PerceptionFrame",
    "TextRecognizer",
    "TextRegionDetector",
    "UiDetector",
]
