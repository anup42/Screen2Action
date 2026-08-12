"""Dataset adapter interfaces and registered public adapters."""

from screen2action.data.adapters.amex import AmexAdapter
from screen2action.data.adapters.android_control import AndroidControlAdapter
from screen2action.data.adapters.base import (
    AdapterAudit,
    AdapterReject,
    CanonicalExample,
    DatasetAdapter,
)
from screen2action.data.adapters.factory import ADAPTERS, adapter_from_json
from screen2action.data.adapters.grounding import (
    RicoSemanticsAdapter,
    ScreenSpotAdapter,
    WaveUiAdapter,
)
from screen2action.data.adapters.guicourse import GuiActAdapter, GuiEnvAdapter

__all__ = [
    "ADAPTERS",
    "AdapterAudit",
    "AdapterReject",
    "AmexAdapter",
    "AndroidControlAdapter",
    "CanonicalExample",
    "DatasetAdapter",
    "GuiActAdapter",
    "GuiEnvAdapter",
    "RicoSemanticsAdapter",
    "ScreenSpotAdapter",
    "WaveUiAdapter",
    "adapter_from_json",
]
