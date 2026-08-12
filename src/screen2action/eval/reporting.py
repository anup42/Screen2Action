"""Atomic evaluation, prediction, and validation-calibration artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from screen2action.config import ResolvedConfig
from screen2action.eval.calibration import (
    calibration_metrics,
    fit_temperature,
    fit_threshold_policy,
)
from screen2action.eval.records import EvaluationRecord
from screen2action.training.artifacts import repository_code_identity


def sha256_file(path: Path) -> str:
    """Hash one artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _environment_snapshot(device: str) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "requested_device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if device == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        snapshot["cuda"] = {
            "runtime": torch.version.cuda,
            "device_index": index,
            "device_name": properties.name,
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": int(properties.total_memory),
        }
    return snapshot


@dataclass(frozen=True, slots=True)
class EvaluationArtifactSet:
    """Paths and immutable identities emitted by one evaluation."""

    output_directory: str
    metrics_json: str
    predictions_csv: str
    predictions_parquet: str
    report_markdown: str
    provenance_json: str
    examples: int
    checkpoint_sha256: str
    data_manifest_digest: str


def _report_markdown(metrics: Mapping[str, Any], provenance: Mapping[str, object]) -> str:
    cascade = metrics["cascade"]
    actions = metrics["actions"]
    structure = metrics["structure"]
    confidence = metrics.get("confidence")
    lines = [
        "# Screen2Action evaluation report",
        "",
        "All cascade rates below are aggregated from per-example observations. They are not ",
        "products of independently estimated stage probabilities.",
        "",
        "## Primary metrics",
        "",
        f"- Examples: {metrics['examples']}",
        f"- Target proposal recall: {cascade['target_proposal_recall']:.6f}",
        "- Target survival given a matched proposal: "
        f"{cascade['target_survival_conditional_on_matched_proposal']:.6f}",
        "- Candidate selection accuracy when target is present: "
        f"{cascade['candidate_selection_accuracy_when_target_present']:.6f}",
        "- Point accuracy when the candidate is correct: "
        f"{cascade['point_accuracy_when_candidate_correct']:.6f}",
        "- End-to-end point-in-target accuracy: "
        f"{cascade['point_in_target_accuracy_end_to_end']:.6f}",
        f"- Action accuracy: {actions['accuracy']:.6f}",
        f"- SSB length median / p90: {structure['ssb_length_median']:.1f} / "
        f"{structure['ssb_length_p90']:.1f}",
        "",
        "## Confidence",
        "",
    ]
    if confidence is None:
        lines.append("No confidence outputs were available.")
    else:
        lines.extend(
            [
                f"- ECE: {confidence['ece']:.6f}",
                f"- Brier: {confidence['brier']:.6f}",
                f"- NLL: {confidence['nll']:.6f}",
            ]
        )
    raw_identity = provenance["identity"]
    if not isinstance(raw_identity, dict):
        raise ValueError("evaluation provenance identity is malformed")
    identity = cast(dict[str, object], raw_identity)
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Checkpoint SHA256: `{identity['checkpoint_sha256']}`",
            f"- Data manifest digest: `{identity['data_manifest_digest']}`",
            f"- Config SHA256: `{identity['config_sha256']}`",
            f"- Code revision: `{identity['code_revision']}`",
            "",
            "See `metrics.json`, `predictions.csv`, `predictions.parquet`, and ",
            "`provenance.json` for complete machine-readable evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_artifacts(
    output_directory: Path,
    *,
    records: Sequence[EvaluationRecord],
    metrics: Mapping[str, Any],
    config: ResolvedConfig,
    checkpoint: Path,
    data_manifest_digest: str,
    cache_manifest_digest: str,
    root: Path,
    device: str,
    evaluation_policy: Mapping[str, object],
) -> EvaluationArtifactSet:
    """Write all required evaluation artifacts and their provenance."""

    if not records:
        raise ValueError("cannot write an empty evaluation")
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "metrics.json",
        "predictions.csv",
        "predictions.parquet",
        "report.md",
        "provenance.json",
        "resolved-config.json",
    )
    existing = [name for name in artifact_names if (output_directory / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite evaluation artifacts: {existing}")
    checkpoint_digest = sha256_file(checkpoint)
    code = repository_code_identity(root)
    provenance: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "identity": {
            "checkpoint_name": checkpoint.name,
            "checkpoint_sha256": checkpoint_digest,
            "data_manifest_digest": data_manifest_digest,
            "cache_manifest_digest": cache_manifest_digest,
            "config_sha256": config.sha256,
            "code_revision": code["git_revision"],
            "working_tree_sha256": code["working_tree_sha256"],
            "code_dirty": code["dirty"],
        },
        "evaluation_policy": dict(evaluation_policy),
        "environment": _environment_snapshot(device),
    }

    metrics_path = output_directory / "metrics.json"
    csv_path = output_directory / "predictions.csv"
    parquet_path = output_directory / "predictions.parquet"
    report_path = output_directory / "report.md"
    provenance_path = output_directory / "provenance.json"
    resolved_path = output_directory / "resolved-config.json"
    _atomic_json(metrics_path, dict(metrics))
    _atomic_json(provenance_path, provenance)
    _atomic_json(resolved_path, config.snapshot())

    rows = [record.as_csv_row() for record in records]
    temporary_csv = csv_path.with_name(f".{csv_path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_csv, csv_path)
    finally:
        temporary_csv.unlink(missing_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("evaluation Parquet output requires: pip install -e .[data]") from error
    temporary_parquet = parquet_path.with_name(f".{parquet_path.name}.{uuid.uuid4().hex}.partial")
    try:
        table = pa.Table.from_pylist(rows)
        parquet.write_table(table, temporary_parquet, compression="zstd")
        os.replace(temporary_parquet, parquet_path)
    finally:
        temporary_parquet.unlink(missing_ok=True)
    _atomic_text(report_path, _report_markdown(metrics, provenance))
    return EvaluationArtifactSet(
        output_directory.as_posix(),
        metrics_path.as_posix(),
        csv_path.as_posix(),
        parquet_path.as_posix(),
        report_path.as_posix(),
        provenance_path.as_posix(),
        len(records),
        checkpoint_digest,
        data_manifest_digest,
    )


def _read_prediction_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    if path.suffix.casefold() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError(
                "calibration Parquet input requires: pip install -e .[data]"
            ) from error
        return [dict(row) for row in parquet.read_table(path).to_pylist()]
    raise ValueError("calibration predictions must be CSV or Parquet")


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value in predictions: {value!r}")


def fit_validation_calibration_artifacts(
    predictions: Path,
    output_directory: Path,
    *,
    target_risk: float = 0.10,
) -> dict[str, object]:
    """Fit temperature and threshold artifacts from held-out non-ScreenSpot validation."""

    rows = _read_prediction_rows(predictions)
    if not rows:
        raise ValueError("calibration predictions are empty")
    if any(str(row.get("split", "")) != "val" for row in rows):
        raise ValueError("confidence calibration requires only split=val predictions")
    if any(str(row.get("source_dataset", "")) == "screenspot" for row in rows):
        raise ValueError("ScreenSpot must never be used for confidence calibration")
    logits = torch.tensor(
        [float(str(row["confidence_logit"])) for row in rows], dtype=torch.float64
    )
    labels = torch.tensor(
        [_bool(row["point_correct_end_to_end"]) for row in rows], dtype=torch.bool
    )
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("calibration logits must be finite")
    provenance_path = predictions.parent / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("evaluation provenance.json is required for calibration")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or not isinstance(provenance.get("identity"), dict):
        raise ValueError("evaluation provenance is malformed")
    identity = dict(provenance["identity"])
    temperature = fit_temperature(logits, labels)
    calibrated = torch.sigmoid(logits / temperature)
    policy = fit_threshold_policy(calibrated, labels, target_risk=target_risk)
    before = calibration_metrics(logits, labels)
    after = calibration_metrics(logits, labels, temperature=temperature)
    binding = {
        "checkpoint_sha256": str(identity.get("checkpoint_sha256", "")),
        "validation_manifest_digest": str(identity.get("data_manifest_digest", "")),
        "validation_predictions_sha256": sha256_file(predictions),
        "examples": len(rows),
        "split": "val",
        "screenspot_excluded": True,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    temperature_path = output_directory / "temperature.json"
    threshold_path = output_directory / "threshold-policy.json"
    report_path = output_directory / "calibration-report.md"
    for path in (temperature_path, threshold_path, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite calibration artifact: {path}")
    _atomic_json(
        temperature_path,
        {
            "schema_version": 1,
            "method": "binary_temperature_scaling",
            "temperature": temperature,
            "before": asdict(before),
            "after": asdict(after),
            "binding": binding,
        },
    )
    _atomic_json(
        threshold_path,
        {
            "schema_version": 1,
            "method": "maximum_validation_coverage_under_empirical_risk",
            "policy": asdict(policy),
            "temperature_artifact_sha256": sha256_file(temperature_path),
            "binding": binding,
        },
    )
    _atomic_text(
        report_path,
        "\n".join(
            [
                "# Confidence calibration report",
                "",
                "Fitted only on held-out validation predictions; ScreenSpot is excluded.",
                "",
                f"- Examples: {len(rows)}",
                f"- Temperature: {temperature:.8f}",
                f"- NLL before / after: {before.nll:.6f} / {after.nll:.6f}",
                f"- ECE before / after: {before.ece:.6f} / {after.ece:.6f}",
                f"- Threshold: {policy.threshold:.8f}",
                f"- Validation coverage / risk: {policy.coverage:.6f} / {policy.observed_risk:.6f}",
                "",
            ]
        ),
    )
    return {
        "status": "written",
        "temperature": temperature,
        "threshold": policy.threshold,
        "coverage": policy.coverage,
        "observed_risk": policy.observed_risk,
        "temperature_artifact": temperature_path.as_posix(),
        "threshold_artifact": threshold_path.as_posix(),
        "report": report_path.as_posix(),
        "binding": binding,
    }
