"""CPU-first Screen2Action reference implementation."""

from screen2action.data.schema import (
    ActionType,
    EdgeRecord,
    ElementAnnotation,
    NodeRecord,
    NodeType,
)

__version__ = "0.1.0"

PACKAGE_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
DATA_SCHEMA_VERSION = 1
MODEL_LOCK_SCHEMA_VERSION = 1
PERCEPTION_CACHE_SCHEMA_VERSION = 1
HANDOFF_SCHEMA_VERSION = 1

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DATA_SCHEMA_VERSION",
    "HANDOFF_SCHEMA_VERSION",
    "MODEL_LOCK_SCHEMA_VERSION",
    "PACKAGE_SCHEMA_VERSION",
    "PERCEPTION_CACHE_SCHEMA_VERSION",
    "ActionType",
    "EdgeRecord",
    "ElementAnnotation",
    "NodeRecord",
    "NodeType",
    "__version__",
]
