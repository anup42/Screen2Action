from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from screen2action.config import ResolvedConfig
from screen2action.eval.reporting import fit_validation_calibration_artifacts, sha256_file
from screen2action.eval.runner import EvaluationSettings, _load_calibration


def _predictions(path: Path, *, split: str, source: str) -> None:
    rows = [
        {
            "command_id": f"c{index}",
            "split": split,
            "source_dataset": source,
            "confidence_logit": logit,
            "point_correct_end_to_end": correct,
        }
        for index, (logit, correct) in enumerate(
            ((-1.0, False), (0.2, True), (1.0, True), (2.0, True))
        )
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (path.parent / "provenance.json").write_text(
        json.dumps(
            {
                "identity": {
                    "checkpoint_sha256": "a" * 64,
                    "data_manifest_digest": "b" * 64,
                }
            }
        ),
        encoding="utf-8",
    )


def test_validation_calibration_writes_separately_bound_artifacts(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    _predictions(predictions, split="val", source="wave_ui")
    result = fit_validation_calibration_artifacts(
        predictions,
        tmp_path / "calibration",
        target_risk=0.25,
    )

    temperature = json.loads(Path(str(result["temperature_artifact"])).read_text(encoding="utf-8"))
    threshold = json.loads(Path(str(result["threshold_artifact"])).read_text(encoding="utf-8"))
    assert temperature["binding"]["split"] == "val"
    assert temperature["binding"]["screenspot_excluded"] is True
    assert threshold["temperature_artifact_sha256"] == sha256_file(
        Path(str(result["temperature_artifact"]))
    )
    assert threshold["policy"]["coverage"] > 0.0
    assert _load_calibration(tmp_path / "calibration", checkpoint_digest="a" * 64) is not None
    threshold["binding"]["checkpoint_sha256"] = "c" * 64
    Path(str(result["threshold_artifact"])).write_text(json.dumps(threshold), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold.*binding"):
        _load_calibration(tmp_path / "calibration", checkpoint_digest="a" * 64)
    threshold["binding"]["checkpoint_sha256"] = "a" * 64
    threshold["temperature_artifact_sha256"] = "d" * 64
    Path(str(result["threshold_artifact"])).write_text(json.dumps(threshold), encoding="utf-8")
    with pytest.raises(ValueError, match="temperature digest"):
        _load_calibration(tmp_path / "calibration", checkpoint_digest="a" * 64)


@pytest.mark.parametrize("k", [1, 4, 8, 16])
def test_evaluation_honors_grounding_k_independently_of_recall_diagnostics(k: int) -> None:
    config = ResolvedConfig({"schema_version": 1, "evaluation": {"retrieval_top_k": [k]}}, (), ())
    settings = EvaluationSettings.from_config(config)
    assert settings.top_k == k
    assert {1, 4, 8}.issubset(settings.recall_ks)


@pytest.mark.parametrize(
    ("split", "source", "message"),
    (("test", "wave_ui", "split=val"), ("val", "screenspot", "ScreenSpot")),
)
def test_calibration_rejects_non_validation_and_screenspot(
    tmp_path: Path,
    split: str,
    source: str,
    message: str,
) -> None:
    predictions = tmp_path / "predictions.csv"
    _predictions(predictions, split=split, source=source)
    with pytest.raises(ValueError, match=message):
        fit_validation_calibration_artifacts(predictions, tmp_path / "calibration")
