from __future__ import annotations

from pathlib import Path

import pytest

from screen2action.data.assets import HuggingFaceDatasetProvider
from screen2action.data.registry import load_source_registry


@pytest.mark.network
@pytest.mark.parametrize(
    "source_name",
    ("guicourse_guiact", "guicourse_guienv", "amex", "wave_ui"),
)
def test_official_huggingface_dataset_revision_resolves(source_name: str) -> None:
    registry = load_source_registry(Path("configs/data/sources.yaml"))
    source = registry.source(source_name)
    resolved = HuggingFaceDatasetProvider().resolve_revision(source, source.revision)
    assert resolved == source.revision
