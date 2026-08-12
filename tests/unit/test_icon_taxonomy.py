"""Public reconstructed icon taxonomy review and freeze gates."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from screen2action.cli import main
from screen2action.perception.taxonomy import (
    LabelObservation,
    build_icon_taxonomy,
    freeze_icon_taxonomy,
    require_frozen_icon_taxonomy,
)


def _observations(count: int = 87) -> tuple[LabelObservation, ...]:
    return tuple(
        LabelObservation("licensed_fixture", f"icon {index:02d}", index + 1)
        for index in range(count)
    )


def test_taxonomy_freeze_requires_exact_count_explicit_acceptance_and_valid_digest() -> None:
    proposal = build_icon_taxonomy(_observations())
    assert proposal["status"] == "ready_for_freeze"
    assert len(proposal["classes"]) == 87

    with pytest.raises(PermissionError, match="accept-reconstruction"):
        freeze_icon_taxonomy(proposal, accept_reconstruction=False)

    frozen = freeze_icon_taxonomy(
        proposal,
        accept_reconstruction=True,
        reviewer="unit-test-reviewer",
    )
    assert frozen["count"] == 87
    assert [item["id"] for item in frozen["classes"]] == list(range(87))
    assert frozen["human_review"]["reviewer"] == "unit-test-reviewer"
    assert require_frozen_icon_taxonomy(frozen) == frozen["digest"]

    tampered = {**frozen, "unknown_policy": "silently changed"}
    with pytest.raises(ValueError, match="digest mismatch"):
        require_frozen_icon_taxonomy(tampered)


def test_taxonomy_does_not_silently_resolve_or_truncate_collisions() -> None:
    observations = (
        LabelObservation("rico", "Arrow Left", 10),
        LabelObservation("other", "arrow-left", 9),
        *_observations(90),
    )
    proposal = build_icon_taxonomy(observations)

    assert proposal["status"] == "review_required"
    assert proposal["observed_count"] == 91
    assert "arrow_left" in proposal["unresolved_alias_collisions"]
    assert len(proposal["classes"]) == 91
    with pytest.raises(ValueError, match="not ready"):
        freeze_icon_taxonomy(proposal, accept_reconstruction=True)


def test_taxonomy_cli_builds_review_report_and_freezes(tmp_path: Path) -> None:
    source = tmp_path / "labels.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "source": "licensed_fixture",
                "labels": {f"icon {index:02d}": index + 1 for index in range(87)},
            }
        ),
        encoding="utf-8",
    )
    proposal = tmp_path / "proposal.yaml"
    frozen = tmp_path / "frozen.yaml"

    assert (
        main(
            [
                "taxonomy",
                "icons",
                "build",
                "--input",
                str(source),
                "--output",
                str(proposal),
                "--json",
            ]
        )
        == 0
    )
    assert proposal.is_file()
    assert proposal.with_suffix(".yaml.review.json").is_file()
    assert (
        main(
            [
                "taxonomy",
                "icons",
                "freeze",
                "--input",
                str(proposal),
                "--output",
                str(frozen),
                "--count",
                "87",
                "--accept-reconstruction",
                "--reviewer",
                "unit-test-reviewer",
                "--json",
            ]
        )
        == 0
    )
    assert require_frozen_icon_taxonomy(yaml.safe_load(frozen.read_text(encoding="utf-8")))


def test_training_gate_rejects_an_unfrozen_proposal() -> None:
    with pytest.raises(ValueError, match="training requires"):
        require_frozen_icon_taxonomy(build_icon_taxonomy(_observations()))
