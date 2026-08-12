"""Opt-in immutable Hub metadata smoke; excluded from default tests."""

from __future__ import annotations

import pytest

from screen2action.model_assets import HuggingFaceProvider, load_model_registry

pytestmark = pytest.mark.network


def test_compact_bert_hub_resolution_is_immutable() -> None:
    registry = load_model_registry("configs/models/registry.yaml")
    spec = registry.roles["command_encoder_bert_l6_h256_a4"]

    resolution = HuggingFaceProvider().resolve(spec)

    assert len(resolution.revision) == 40
    assert resolution.files
    assert any(file.remote_path.endswith((".bin", ".safetensors")) for file in resolution.files)
