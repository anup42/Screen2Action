"""Configuration-driven B/K and ablation evaluation orchestration."""

from __future__ import annotations

import csv
import json
import os
import re
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from screen2action.config import ResolvedConfig
from screen2action.eval.runner import run_evaluation

STANDARD_ABLATIONS = (
    "learned_selector",
    "confidence_only_selector",
    "no_target_survival_loss",
    "no_reference_survival_loss",
    "no_budget_loss",
    "no_relation_reranking",
    "no_graph_relations",
    "paper_equation_graph_reranker",
    "cpu_reconstruction_graph_reranker",
    "mobilevit_accurate",
    "roi_align_fast",
    "quantization_fp32",
    "quantization_ptq_weight_only",
    "quantization_qat_fake_quant",
)
_SEPARATE_CHECKPOINT = {
    "no_target_survival_loss",
    "no_reference_survival_loss",
    "no_budget_loss",
    "quantization_qat_fake_quant",
}


@dataclass(frozen=True, slots=True)
class SweepJob:
    """One auditable evaluation job and its checkpoint-lineage requirement."""

    name: str
    family: str
    config_updates: Mapping[str, object]
    checkpoint_variant: str | None = None
    supported: bool = True
    reason: str = ""


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return cast(dict[str, object], value)


def _positive_values(value: object, *, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{context} must contain positive integers")
        result.append(item)
    return tuple(sorted(set(result)))


def build_sweep_jobs(config: ResolvedConfig) -> tuple[SweepJob, ...]:
    """Expand the required matrix while keeping training ablations honest."""

    evaluation = _mapping(config.values.get("evaluation"), context="evaluation")
    budgets = _positive_values(
        evaluation.get("budgets", [128, 256, 512]),
        context="evaluation.budgets",
    )
    top_ks = _positive_values(
        evaluation.get("retrieval_top_k", [4, 8, 16]),
        context="evaluation.retrieval_top_k",
    )
    raw_ablations = evaluation.get("ablations", list(STANDARD_ABLATIONS))
    if not isinstance(raw_ablations, list) or not all(
        isinstance(value, str) for value in raw_ablations
    ):
        raise ValueError("evaluation.ablations must be a list of names")
    ablations = tuple(cast(list[str], raw_ablations))
    unknown = sorted(set(ablations) - set(STANDARD_ABLATIONS))
    if unknown:
        raise ValueError(f"unknown evaluation ablations: {unknown}")
    jobs = [
        SweepJob(
            f"budget-{budget}-k-{top_k}",
            "budget_k",
            {
                "evaluation.budgets": [budget],
                "evaluation.retrieval_top_k": [top_k],
                "evaluation.selector_variant": "learned",
                "evaluation.graph_relations": "all",
                "evaluation.relation_reranking": True,
                "evaluation.quantization_variant": "fp32",
            },
        )
        for budget in budgets
        for top_k in top_ks
    ]
    ablation_updates: dict[str, Mapping[str, object]] = {
        "learned_selector": {"evaluation.selector_variant": "learned"},
        "confidence_only_selector": {"evaluation.selector_variant": "confidence_only"},
        "no_target_survival_loss": {},
        "no_reference_survival_loss": {},
        "no_budget_loss": {},
        "no_relation_reranking": {"evaluation.relation_reranking": False},
        "no_graph_relations": {"evaluation.graph_relations": "none"},
        "paper_equation_graph_reranker": {
            "model.graph_variant": "paper_eq_v1",
            "evaluation.graph_relations": "all",
            "evaluation.relation_reranking": True,
        },
        "cpu_reconstruction_graph_reranker": {
            "model.graph_variant": "cpu_reconstruction_v1",
            "evaluation.graph_relations": "all",
            "evaluation.relation_reranking": True,
        },
        "mobilevit_accurate": {"evaluation.visual_variant": "mobilevit_accurate"},
        "roi_align_fast": {"evaluation.visual_variant": "roi_align_fast"},
        "quantization_fp32": {"evaluation.quantization_variant": "fp32"},
        "quantization_ptq_weight_only": {"evaluation.quantization_variant": "ptq_weight_only"},
        "quantization_qat_fake_quant": {"evaluation.quantization_variant": "qat_fake_quant"},
    }
    for name in ablations:
        supported = name != "roi_align_fast"
        jobs.append(
            SweepJob(
                name,
                "ablation",
                ablation_updates[name],
                checkpoint_variant=name if name in _SEPARATE_CHECKPOINT else None,
                supported=supported,
                reason=(
                    "optional ROIAlign Fast crop encoder is not implemented; no artifact or "
                    "metric is substituted"
                    if not supported
                    else ""
                ),
            )
        )
    return tuple(jobs)


def _set_dotted(values: dict[str, Any], key: str, value: object) -> None:
    parts = key.split(".")
    cursor = values
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"sweep update crosses non-mapping configuration key: {key}")
        cursor = cast(dict[str, Any], child)
    cursor[parts[-1]] = value


def _job_config(config: ResolvedConfig, job: SweepJob) -> ResolvedConfig:
    values = cast(dict[str, Any], deepcopy(dict(config.values)))
    evaluation = values.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
        values["evaluation"] = evaluation
    evaluation["mode"] = "single"
    for key, value in job.config_updates.items():
        _set_dotted(values, key, value)
    return ResolvedConfig(
        cast(Any, values),
        source_paths=config.source_paths,
        overrides=(*config.overrides, f"sweep.job={job.name}"),
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metric(metrics: Mapping[str, object], *keys: str) -> float | None:
    value: object = metrics
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, int | float) else None


def run_evaluation_sweep(
    config: ResolvedConfig,
    *,
    checkpoint: Path,
    manifest: Path,
    cache_manifest: Path,
    output_directory: Path,
    model_lock: Path | None = None,
    model_cache_root: Path | None = None,
    device: str = "cpu",
    split: str | None = None,
    max_commands: int | None = None,
    calibration: Path | None = None,
) -> dict[str, object]:
    """Execute runnable jobs and report explicit external-checkpoint/optional gaps."""

    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite a non-empty sweep directory: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluation = _mapping(config.values.get("evaluation"), context="evaluation")
    variants = _mapping(
        evaluation.get("checkpoint_variants"),
        context="evaluation.checkpoint_variants",
    )
    rows: list[dict[str, object]] = []
    for job in build_sweep_jobs(config):
        row: dict[str, object] = {
            "name": job.name,
            "family": job.family,
            "settings": dict(job.config_updates),
        }
        if not job.supported:
            rows.append({**row, "status": "unsupported_optional", "reason": job.reason})
            continue
        selected_checkpoint = checkpoint
        if job.checkpoint_variant is not None:
            raw_path = variants.get(job.checkpoint_variant)
            if not isinstance(raw_path, str) or not raw_path:
                rows.append(
                    {
                        **row,
                        "status": "requires_checkpoint",
                        "reason": (
                            "training-loss or QAT ablations require their own checkpoint; "
                            f"set evaluation.checkpoint_variants.{job.checkpoint_variant}"
                        ),
                    }
                )
                continue
            selected_checkpoint = Path(raw_path)
            if not selected_checkpoint.is_file():
                rows.append(
                    {
                        **row,
                        "status": "requires_checkpoint",
                        "reason": f"configured checkpoint is unavailable: {selected_checkpoint}",
                    }
                )
                continue
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", job.name)
        result = run_evaluation(
            _job_config(config, job),
            checkpoint=selected_checkpoint,
            manifest=manifest,
            cache_manifest=cache_manifest,
            output_directory=output_directory / "jobs" / safe_name,
            model_lock=model_lock,
            model_cache_root=model_cache_root,
            device=device,
            split=split,
            max_commands=max_commands,
            calibration=calibration,
        )
        metrics = result.metrics
        rows.append(
            {
                **row,
                "status": "complete",
                "checkpoint": selected_checkpoint.as_posix(),
                "output_directory": result.artifact_set.output_directory,
                "proposal_recall": _metric(metrics, "cascade", "target_proposal_recall"),
                "retrieval_recall": _metric(
                    metrics,
                    "cascade",
                    "retrieval_recall",
                    str(result.settings.top_k),
                    "end_to_end",
                ),
                "point_in_target_accuracy": _metric(
                    metrics,
                    "cascade",
                    "point_in_target_accuracy_end_to_end",
                ),
                "action_accuracy": _metric(metrics, "actions", "accuracy"),
            }
        )
    json_path = output_directory / "sweep.json"
    csv_path = output_directory / "sweep.csv"
    report_path = output_directory / "sweep.md"
    payload = {
        "schema_version": 1,
        "jobs": rows,
        "complete": sum(row["status"] == "complete" for row in rows),
        "requires_checkpoint": sum(row["status"] == "requires_checkpoint" for row in rows),
        "unsupported_optional": sum(row["status"] == "unsupported_optional" for row in rows),
        "metric_separation": "each job retains its own artifact directory",
    }
    _atomic_text(json_path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    flat_rows = [
        {
            **{key: value for key, value in row.items() if key != "settings"},
            "settings": json.dumps(row["settings"], sort_keys=True, separators=(",", ":")),
        }
        for row in rows
    ]
    temporary_csv = csv_path.with_name(f".{csv_path.name}.{uuid.uuid4().hex}.partial")
    try:
        fieldnames = sorted({key for row in flat_rows for key in row})
        with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)
        os.replace(temporary_csv, csv_path)
    finally:
        temporary_csv.unlink(missing_ok=True)
    lines = [
        "# Screen2Action evaluation sweep",
        "",
        "Metrics are never merged across Accurate/Fast, graph, checkpoint, "
        "or quantization variants.",
        "",
        "| Job | Family | Status | Point-in-target | Reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        point = row.get("point_in_target_accuracy")
        display = f"{point:.6f}" if isinstance(point, float) else "-"
        lines.append(
            f"| {row['name']} | {row['family']} | {row['status']} | {display} | "
            f"{row.get('reason', '')} |"
        )
    lines.append("")
    _atomic_text(report_path, "\n".join(lines))
    return {
        "status": "complete_with_explicit_gaps"
        if payload["requires_checkpoint"] or payload["unsupported_optional"]
        else "complete",
        "output_directory": output_directory.as_posix(),
        "summary_json": json_path.as_posix(),
        "summary_csv": csv_path.as_posix(),
        "report": report_path.as_posix(),
        **{
            key: payload[key] for key in ("complete", "requires_checkpoint", "unsupported_optional")
        },
    }
