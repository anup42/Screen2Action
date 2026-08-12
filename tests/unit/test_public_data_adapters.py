from __future__ import annotations

from pathlib import Path

import pytest

from screen2action.data.adapters.factory import ADAPTERS, adapter_from_json
from screen2action.data.registry import load_source_registry
from screen2action.data.schema import PointSource

FIXTURES = Path(__file__).parents[1] / "fixtures" / "data"


@pytest.mark.parametrize("source", sorted(ADAPTERS))
def test_registered_source_fixture_converts_without_geometry_coercion(source: str) -> None:
    adapter = adapter_from_json(
        source,
        FIXTURES / f"{source}.json",
        revision="fixture-revision",
        license_acknowledgement_id="fixture-license-ack",
    )
    assert adapter.examples()
    for example in adapter.examples():
        assert example.screen.provenance.source_dataset == source
        assert example.screen.provenance.source_revision == "fixture-revision"
        assert example.screen.provenance.source_license_ack_id == "fixture-license-ack"
        for element in example.screen.elements:
            x1, y1, x2, y2 = element.box_xyxy_norm
            assert 0.0 <= x1 <= x2 <= 1.0
            assert 0.0 <= y1 <= y2 <= 1.0


def test_true_and_pseudo_points_remain_distinct() -> None:
    gui_act = adapter_from_json(
        "guicourse_guiact",
        FIXTURES / "guicourse_guiact.json",
        revision="fixture",
        license_acknowledgement_id="ack",
    )
    gui_env = adapter_from_json(
        "guicourse_guienv",
        FIXTURES / "guicourse_guienv.json",
        revision="fixture",
        license_acknowledgement_id="ack",
    )
    assert gui_act.commands()[0].target_point_source is PointSource.TRUE
    assert gui_act.commands()[0].label_masks["point"]
    assert gui_env.commands()[0].target_point_source is PointSource.PSEUDO_BOX_CENTER
    assert not gui_env.commands()[0].label_masks["point"]


def test_unsupported_steps_and_missing_row_license_are_audited() -> None:
    amex = adapter_from_json(
        "amex",
        FIXTURES / "amex.json",
        revision="fixture",
        license_acknowledgement_id="ack",
    )
    wave = adapter_from_json(
        "wave_ui",
        FIXTURES / "wave_ui.json",
        revision="fixture",
        license_acknowledgement_id="ack",
    )
    assert amex.audit().rejection_counts == {"unsupported_action": 1}
    assert wave.audit().rejection_counts == {"missing_row_license_source": 1}


def test_source_registry_pins_all_sources_and_guards_screenspot() -> None:
    registry = load_source_registry(Path("configs/data/sources.yaml"))
    assert set(registry.sources) == set(ADAPTERS)
    assert all(source.revision != "unresolved" for source in registry.sources.values())
    screenspot = registry.source("screenspot")
    assert screenspot.evaluation_only
    assert screenspot.hard_leakage_guard
    assert not screenspot.train_eligible
