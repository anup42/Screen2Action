"""Typed real, cached, and oracle perception interfaces.

Cache orchestration is lazy so importing data-manifest validation cannot cycle
back through precompute, which itself validates the data manifest.
"""

from __future__ import annotations

import importlib
from typing import Any

from screen2action.perception.base import TextRecognizer, TextRegionDetector, UiDetector
from screen2action.perception.cache import ContentAddressedPerceptionCache, PerceptionCacheKey
from screen2action.perception.oracle import OracleFrame, OraclePerception
from screen2action.perception.pipeline import (
    FullScreenPerception,
    FullScreenPerceptionConfig,
    PerceptionFrame,
)

_LAZY_EXPORTS = {
    "CachedPerceptionLoader": ("screen2action.perception.loader", "CachedPerceptionLoader"),
    "PerceptionBundleSpec": ("screen2action.perception.precompute", "PerceptionBundleSpec"),
    "precompute_perception": ("screen2action.perception.precompute", "precompute_perception"),
    "validate_perception_cache": (
        "screen2action.perception.precompute",
        "validate_perception_cache",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


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
