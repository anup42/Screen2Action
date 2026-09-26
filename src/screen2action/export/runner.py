"""Atomic, profile-separated ONNX export with mandatory CPU parity reports."""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import torch
import yaml
from torch import nn

from screen2action.config import ResolvedConfig
from screen2action.data.schema import NodeType
from screen2action.export.host_contract import validate_fixed_graph_inputs
from screen2action.export.onnx import export_onnx
from screen2action.export.parity import onnx_parity
from screen2action.export.partitions import (
    COMMAND_INPUT_NAMES,
    COMMAND_OUTPUT_NAMES,
    GRAPH_INPUT_NAMES,
    GRAPH_OUTPUT_NAMES,
    GROUNDING_INPUT_NAMES,
    GROUNDING_OUTPUT_NAMES,
    VISUAL_INPUT_NAMES,
    VISUAL_OUTPUT_NAMES,
    CommandRetrievalPartition,
    CropGroundingPartition,
    GraphRetentionPartition,
    VisualEncoderPartition,
)
from screen2action.export.static_shapes import ExportShapeContract
from screen2action.handoff import repository_root
from screen2action.model_assets import sha256_file
from screen2action.perception.factory import load_visual_checkpoint_weights
from screen2action.training.artifacts import repository_code_identity
from screen2action.training.checkpoints import load_model_weights
from screen2action.training.model_factory import build_screen2action_model
from screen2action.training.semantics import build_semantics_graph_model

EXPORT_PROFILES = ("accurate", "fast_roi", "tiny_cpu")
PARTITION_NAMES = (
    "01_visual_encoder",
    "02_graph_retention",
    "03_command_retrieval_reranking",
    "04_crop_grounding_actions",
)

HOST_OPERATIONS = (
    {
        "operation": "image_decode_resize_normalize",
        "owner": "host",
        "reason": "backend-specific image and model preprocessing",
    },
    {
        "operation": "detector_nms_and_ocr_decode",
        "owner": "host",
        "reason": "discrete external perception postprocessing",
    },
    {
        "operation": "canonical_graph_construction",
        "owner": "host",
        "reason": "deterministic normalized-xyxy relation policy",
    },
    {
        "operation": "budget_select_and_closure_repair",
        "owner": "host",
        "reason": "exact independent knapsack plus deterministic repair",
    },
    {
        "operation": "actionability_filter_and_fixed_top_k",
        "owner": "host",
        "reason": "deterministic candidate identity and padding",
    },
    {
        "operation": "crop_expansion_and_extraction",
        "owner": "host",
        "reason": "pixel access remains outside neural graphs",
    },
    {
        "operation": "candidate_local_to_screen_coordinate_conversion",
        "owner": "host",
        "reason": "reversible normalized-xyxy boundary conversion",
    },
)


@dataclass(frozen=True, slots=True)
class PartitionedExportResult:
    """Portable summary of one profile-specific export package."""

    ok: bool
    status: str
    profile: str
    output_directory: str
    checkpoint_sha256: str
    exported_partitions: tuple[str, ...]
    unsupported_or_failed_partitions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return cast(dict[str, object], value)


def _integer(values: Mapping[str, object], key: str, default: int) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"export.{key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str, default: float) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"export.{key} must be numeric")
    return float(value)


def _model_integer(values: Mapping[str, object], key: str, default: int) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"model.{key} must be an integer")
    return value


def _contract(config: ResolvedConfig, profile: str) -> ExportShapeContract:
    model = _mapping(config.values.get("model"), context="model")
    export = _mapping(config.values.get("export"), context="export")
    model_profile = str(model.get("profile", ""))
    expected_model_profile = "tiny_cpu" if profile == "tiny_cpu" else "paper_reference"
    if model_profile != expected_model_profile:
        raise ValueError(
            f"export profile {profile} requires model.profile={expected_model_profile}"
        )
    expected_variant = str(export.get("profile", profile))
    if expected_variant != profile:
        raise ValueError("export.profile must match the requested CLI profile")
    paper = expected_model_profile == "paper_reference"
    max_nodes = _integer(export, "max_nodes", 64 if paper else 8)
    return ExportShapeContract(
        max_nodes=max_nodes,
        max_edges=_integer(export, "max_edges", min(max_nodes * max_nodes, 512 if paper else 32)),
        max_command_tokens=_model_integer(model, "command_max_length", 64 if paper else 32),
        top_k=_model_integer(model, "top_k", 8 if paper else 4),
        crop_tokens=_model_integer(model, "crop_tokens", 144 if paper else 9),
        embedding_dim=_model_integer(model, "embedding_dim", 256 if paper else 32),
        batch_size=_integer(export, "batch_size", 1),
        node_text_tokens=_model_integer(model, "node_text_length", 16 if paper else 12),
        icon_class_count=_model_integer(model, "icon_class_count", 87),
        visual_feature_dim=_model_integer(model, "visual_feature_dim", 256 if paper else 16),
        geometry_dim=_integer(export, "geometry_dim", 12),
        crop_size=_model_integer(model, "crop_size", 192 if paper else 48),
        semantic_crop_size=_model_integer(model, "semantic_crop_size", 224 if paper else 64),
        visual_batch_size=_integer(export, "visual_batch_size", 16 if paper else 8),
        action_count=_integer(export, "action_count", 4),
    )


def _box_tensor(*shape: int, generator: torch.Generator) -> torch.Tensor:
    starts = torch.rand((*shape, 2), generator=generator) * 0.65
    extents = 0.08 + torch.rand((*shape, 2), generator=generator) * 0.25
    return torch.cat((starts, (starts + extents).clamp_max(1.0)), dim=-1)


def _dense_graph_inputs(
    contract: ExportShapeContract,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = contract.batch_size
    nodes = contract.max_nodes
    relation_mask = torch.zeros((batch, 3, nodes, nodes), dtype=torch.bool)
    geometry = torch.rand((batch, 3, nodes, nodes, contract.geometry_dim), generator=generator)
    valid = torch.ones((batch, nodes), dtype=torch.bool)
    if nodes > contract.top_k:
        valid[:, -1] = False
    active = max(1, int(valid[0].sum()))
    edge_limit = min(contract.max_edges, 3 * active * max(active - 1, 0))
    written = 0
    for source in range(active):
        for offset in range(1, active):
            destination = (source + offset) % active
            for relation in range(3):
                if written >= edge_limit:
                    break
                relation_mask[:, relation, source, destination] = True
                written += 1
            if written >= edge_limit:
                break
        if written >= edge_limit:
            break
    validate_fixed_graph_inputs(contract, relation_mask, geometry, valid)
    return relation_mask, geometry, valid


def _visual_inputs(
    contract: ExportShapeContract, generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    crops = torch.rand(
        (
            contract.visual_batch_size,
            3,
            contract.semantic_crop_size,
            contract.semantic_crop_size,
        ),
        generator=generator,
    )
    valid = torch.ones(contract.visual_batch_size, dtype=torch.bool)
    if contract.visual_batch_size > 1:
        valid[-1] = False
    return crops, valid


def _graph_inputs(
    contract: ExportShapeContract,
    *,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    batch = contract.batch_size
    nodes = contract.max_nodes
    relation_mask, geometry, valid = _dense_graph_inputs(contract, generator=generator)
    text_ids = torch.randint(
        1,
        max(vocab_size, 2),
        (batch, nodes, contract.node_text_tokens),
        generator=generator,
    )
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)
    text_mask[:, :, -1] = False
    alternating = torch.arange(nodes).remainder(2).bool().unsqueeze(0).expand(batch, -1)
    return (
        torch.randint(0, len(NodeType), (batch, nodes), generator=generator),
        torch.randint(0, 56, (batch, nodes), generator=generator),
        alternating,
        _box_tensor(batch, nodes, generator=generator),
        torch.randint(0, 8, (batch, nodes), generator=generator),
        text_ids,
        text_mask,
        torch.rand((batch, nodes, contract.icon_class_count), generator=generator),
        ~alternating,
        torch.rand((batch, nodes, contract.visual_feature_dim), generator=generator),
        alternating,
        torch.randn((batch, nodes, 4), generator=generator),
        torch.ones((batch, nodes, 4), dtype=torch.bool),
        torch.rand((batch, nodes, 4), generator=generator),
        valid,
        relation_mask,
        geometry,
    )


def _command_inputs(
    contract: ExportShapeContract,
    *,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    relation_mask, geometry, valid = _dense_graph_inputs(contract, generator=generator)
    command_ids = torch.randint(
        1,
        max(vocab_size, 2),
        (contract.batch_size, contract.max_command_tokens),
        generator=generator,
    )
    command_mask = torch.ones_like(command_ids, dtype=torch.bool)
    if contract.max_command_tokens > 1:
        command_mask[:, -1] = False
    graph_states = torch.randn(
        (contract.batch_size, contract.max_nodes, contract.embedding_dim),
        generator=generator,
    )
    return command_ids, command_mask, graph_states, relation_mask, geometry, valid


def _grounding_inputs(
    contract: ExportShapeContract, generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    batch = contract.batch_size
    candidates = contract.top_k
    command_states = torch.randn(
        (batch, contract.max_command_tokens, contract.embedding_dim), generator=generator
    )
    command_mask = torch.ones((batch, contract.max_command_tokens), dtype=torch.bool)
    if contract.max_command_tokens > 1:
        command_mask[:, -1] = False
    candidate_mask = torch.ones((batch, candidates), dtype=torch.bool)
    if candidates > 1:
        candidate_mask[:, -1] = False
    crop_mask = candidate_mask.unsqueeze(-1).expand(-1, -1, contract.crop_tokens).clone()
    return (
        torch.rand(
            (batch, candidates, 3, contract.crop_size, contract.crop_size),
            generator=generator,
        ),
        command_states,
        command_mask,
        torch.randn((batch, candidates, contract.embedding_dim), generator=generator),
        candidate_mask,
        crop_mask,
        _box_tensor(batch, candidates, generator=generator),
    )


def _tensor_specs(names: Sequence[str], tensors: Sequence[torch.Tensor]) -> list[dict[str, object]]:
    if len(names) != len(tensors):
        raise ValueError("tensor names do not match fixed partition outputs")
    return [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        for name, tensor in zip(names, tensors, strict=True)
    ]


def _module_outputs(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    module.eval()
    with torch.no_grad():
        raw = module(*inputs)
    if isinstance(raw, torch.Tensor):
        return (raw,)
    if not isinstance(raw, tuple) or not all(isinstance(value, torch.Tensor) for value in raw):
        raise ValueError("export partition must return a tensor tuple")
    return cast(tuple[torch.Tensor, ...], raw)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _export_partition(
    directory: Path,
    *,
    name: str,
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset_version: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    import onnx

    started = time.perf_counter()
    module.cpu().eval()
    final_path = directory / f"{name}.onnx"
    temporary = directory / f".{name}.partial.onnx"
    try:
        export_onnx(
            module,
            inputs,
            temporary,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
        )
        onnx.checker.check_model(onnx.load(str(temporary)))
        errors = onnx_parity(module, temporary, inputs, atol=atol, rtol=rtol)
        outputs = _module_outputs(module, inputs)
        os.replace(temporary, final_path)
    finally:
        temporary.unlink(missing_ok=True)
    parity = {
        "backend": "onnxruntime_cpu",
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error_by_output": {
            output_name: error for output_name, error in zip(output_names, errors, strict=True)
        },
        "passed": True,
    }
    _write_json(directory / f"{name}.parity.json", parity)
    return {
        "name": name,
        "status": "exported",
        "artifact": final_path.name,
        "artifact_sha256": sha256_file(final_path),
        "artifact_bytes": final_path.stat().st_size,
        "opset_version": opset_version,
        "fixed_shapes": True,
        "inputs": _tensor_specs(input_names, inputs),
        "outputs": _tensor_specs(output_names, outputs),
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        "parity": parity,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _failure(name: str, error: Exception) -> dict[str, object]:
    return {
        "name": name,
        "status": "failed",
        "error_type": type(error).__name__,
        "reason": str(error),
        "behavior_substitution": False,
    }


def _unsupported(name: str, reason: str) -> dict[str, object]:
    return {
        "name": name,
        "status": "unsupported",
        "reason": reason,
        "behavior_substitution": False,
    }


def _checkpoint_metadata(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("export checkpoint payload is malformed")
    return {
        "sha256": sha256_file(path),
        "format_version": int(payload.get("format_version", 0)),
        "source_stage": str(payload.get("stage", payload.get("source_stage", "unknown"))),
        "source_epoch": int(payload.get("epoch", 0)),
        "source_step": int(payload.get("step", 0)),
        "data_manifest_digest": str(payload.get("manifest_hash", "unknown")),
        "model_lock_digest": str(payload.get("model_lock_hash", "unknown")),
        "perception_cache_manifest_digest": str(payload.get("cache_manifest_hash", "unknown")),
        "source_git_revision": str(payload.get("git_revision", "unknown")),
    }


def _package_report(
    *,
    profile: str,
    config: ResolvedConfig,
    contract: ExportShapeContract,
    checkpoint: Mapping[str, object],
    partitions: Sequence[Mapping[str, object]],
    model_lock_digest: str,
    visual_checkpoint: Mapping[str, object] | None,
) -> dict[str, object]:
    statuses = [str(partition["status"]) for partition in partitions]
    ok = all(status == "exported" for status in statuses)
    if ok:
        status = "complete"
    elif all(value == "unsupported" for value in statuses):
        status = "unsupported"
    elif all(value == "failed" for value in statuses):
        status = "failed"
    else:
        status = "partial"
    truth_class = "cpu_reconstruction_export"
    if profile == "accurate":
        truth_class = "paper_reference_locked_export" if ok else "paper_reference_export_attempt"
    elif profile == "fast_roi":
        truth_class = "unsupported_optional_fast_reconstruction"
    package_versions: dict[str, str] = {}
    for package in ("onnx", "onnxruntime", "torch"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "unavailable"
    return {
        "schema_version": 1,
        "ok": ok,
        "status": status,
        "profile": profile,
        "artifact_namespace": profile,
        "metrics_may_merge_with_other_profiles": False,
        "truth_class": truth_class,
        "mobile_runtime_measured": False,
        "quantized_accuracy_measured": False,
        "config_sha256": config.sha256,
        "model_lock_digest": model_lock_digest,
        "checkpoint": dict(checkpoint),
        "visual_checkpoint": dict(visual_checkpoint) if visual_checkpoint is not None else None,
        "shape_contract": contract.as_dict(),
        "host_operations": list(HOST_OPERATIONS),
        "partitions": list(partitions),
        "code": repository_code_identity(repository_root()),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions,
            "onnx_provider": "CPUExecutionProvider",
        },
    }


def run_partitioned_export(
    config: ResolvedConfig,
    *,
    checkpoint: Path,
    output_directory: Path,
    profile: str,
    model_lock: Path | None = None,
    model_cache_root: Path | None = None,
    visual_checkpoint: Path | None = None,
) -> PartitionedExportResult:
    """Export one immutable profile directory and validate it on CPU."""

    if profile not in EXPORT_PROFILES:
        raise ValueError(f"unsupported export profile: {profile}")
    resolved_checkpoint = checkpoint.resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"export checkpoint does not exist: {checkpoint}")
    contract = _contract(config, profile)
    export_values = _mapping(config.values.get("export"), context="export")
    opset_version = _integer(export_values, "opset_version", 17)
    atol = _number(export_values, "parity_atol", 1e-4)
    rtol = _number(export_values, "parity_rtol", 1e-3)
    if opset_version < 17:
        raise ValueError("export.opset_version must be at least 17")
    if atol <= 0.0 or rtol <= 0.0:
        raise ValueError("export parity tolerances must be positive")
    output_root = output_directory.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / profile
    if destination.exists():
        raise FileExistsError(f"profile export directory already exists: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{profile}-", dir=output_root))
    checkpoint_metadata = _checkpoint_metadata(resolved_checkpoint)
    partitions: list[dict[str, object]] = []
    model_lock_digest = str(checkpoint_metadata["model_lock_digest"])
    visual_metadata: dict[str, object] | None = None
    if profile == "fast_roi":
        reason = (
            "detector ROIAlign crop tokens and a distinct Fast-trained checkpoint topology "
            "are not implemented; the Accurate MobileViT path is not substituted"
        )
        partitions.extend(_unsupported(name, reason) for name in PARTITION_NAMES)
    else:
        model_values = _mapping(config.values.get("model"), context="model")
        vocab_size = _model_integer(model_values, "vocab_size", 16_384)
        cache_root = model_cache_root or repository_root() / ".cache" / "models"
        try:
            bundle = build_screen2action_model(
                cast(Mapping[str, object], config.values),
                tokenizer_vocab_size=vocab_size,
                model_lock_path=model_lock,
                cache_root=cache_root,
            )
            model_lock_digest = bundle.model_lock_digest
            load_model_weights(
                resolved_checkpoint,
                bundle.model,
                device="cpu",
                expected_model_lock_hash=bundle.model_lock_digest,
            )
            bundle.model.cpu().eval()
        except Exception as error:
            partitions.extend(_failure(name, error) for name in PARTITION_NAMES)
        else:
            generator = torch.Generator(device="cpu").manual_seed(20260813)
            if visual_checkpoint is None:
                partitions.append(
                    _unsupported(
                        PARTITION_NAMES[0],
                        "a trained Stage-1 visual checkpoint is required; "
                        "untrained heads are not exported",
                    )
                )
            else:
                try:
                    semantics = build_semantics_graph_model(
                        cast(Mapping[str, object], config.values),
                        model_lock_path=model_lock,
                        cache_root=cache_root,
                    )
                    if semantics.model_lock_digest != bundle.model_lock_digest:
                        raise ValueError("visual and downstream model-lock digests differ")
                    visual_metadata = load_visual_checkpoint_weights(
                        semantics.model.visual,
                        visual_checkpoint,
                        expected_model_lock_digest=bundle.model_lock_digest,
                    )
                    partitions.append(
                        _export_partition(
                            stage,
                            name=PARTITION_NAMES[0],
                            module=VisualEncoderPartition(semantics.model.visual),
                            inputs=_visual_inputs(contract, generator),
                            input_names=VISUAL_INPUT_NAMES,
                            output_names=VISUAL_OUTPUT_NAMES,
                            opset_version=opset_version,
                            atol=atol,
                            rtol=rtol,
                        )
                    )
                except Exception as error:
                    partitions.append(_failure(PARTITION_NAMES[0], error))
            definitions: tuple[
                tuple[
                    str,
                    nn.Module,
                    tuple[torch.Tensor, ...],
                    Sequence[str],
                    Sequence[str],
                ],
                ...,
            ] = (
                (
                    PARTITION_NAMES[1],
                    GraphRetentionPartition(
                        bundle.model.node_encoder,
                        bundle.model.graph_encoder,
                        bundle.model.retention_scorer,
                    ),
                    _graph_inputs(contract, vocab_size=vocab_size, generator=generator),
                    GRAPH_INPUT_NAMES,
                    GRAPH_OUTPUT_NAMES,
                ),
                (
                    PARTITION_NAMES[2],
                    CommandRetrievalPartition(
                        bundle.model.command_encoder,
                        bundle.model.retriever,
                        bundle.model.reranker,
                    ),
                    _command_inputs(contract, vocab_size=vocab_size, generator=generator),
                    COMMAND_INPUT_NAMES,
                    COMMAND_OUTPUT_NAMES,
                ),
                (
                    PARTITION_NAMES[3],
                    CropGroundingPartition(bundle.model.crop_encoder, bundle.model.grounder),
                    _grounding_inputs(contract, generator),
                    GROUNDING_INPUT_NAMES,
                    GROUNDING_OUTPUT_NAMES,
                ),
            )
            for name, module, inputs, input_names, output_names in definitions:
                try:
                    partitions.append(
                        _export_partition(
                            stage,
                            name=name,
                            module=module,
                            inputs=inputs,
                            input_names=input_names,
                            output_names=output_names,
                            opset_version=opset_version,
                            atol=atol,
                            rtol=rtol,
                        )
                    )
                except Exception as error:
                    partitions.append(_failure(name, error))
    report = _package_report(
        profile=profile,
        config=config,
        contract=contract,
        checkpoint=checkpoint_metadata,
        partitions=partitions,
        model_lock_digest=model_lock_digest,
        visual_checkpoint=visual_metadata,
    )
    _write_json(stage / "shape-contract.json", contract.as_dict())
    _write_json(
        stage / "host-runtime-contract.json",
        {
            "schema_version": 1,
            "box_format": "normalized_xyxy",
            "operations": list(HOST_OPERATIONS),
        },
    )
    (stage / "resolved-config.yaml").write_text(
        yaml.safe_dump(config.snapshot(), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    _write_json(stage / "partition-report.json", report)
    os.replace(stage, destination)
    exported = tuple(
        str(partition["name"]) for partition in partitions if partition["status"] == "exported"
    )
    incomplete = tuple(
        str(partition["name"]) for partition in partitions if partition["status"] != "exported"
    )
    return PartitionedExportResult(
        ok=bool(report["ok"]),
        status=str(report["status"]),
        profile=profile,
        output_directory=(output_directory / profile).as_posix(),
        checkpoint_sha256=str(checkpoint_metadata["sha256"]),
        exported_partitions=exported,
        unsupported_or_failed_partitions=incomplete,
    )
