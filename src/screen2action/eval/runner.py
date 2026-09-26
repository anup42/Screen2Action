"""Manifest-bound cached-perception evaluation and direct cascade reporting."""

from __future__ import annotations

import json
import math
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch

from screen2action.config import ResolvedConfig
from screen2action.data.schema import ActionType, Box, NodeType, Point
from screen2action.eval.metrics import aggregate_evaluation_records
from screen2action.eval.records import EvaluationRecord
from screen2action.eval.reporting import (
    EvaluationArtifactSet,
    sha256_file,
    write_evaluation_artifacts,
)
from screen2action.eval.screenspot import (
    official_point_in_target,
    require_evaluation_only_screenspot,
    target_type_for_screen,
)
from screen2action.handoff import repository_root
from screen2action.perception.loader import CachedPerceptionLoader
from screen2action.runtime.cropper import crop_to_screen_point
from screen2action.ssb.matching import match_positive_proposal
from screen2action.training.checkpoints import load_model_weights, manifest_hash
from screen2action.training.data import (
    CachedScreenBatchBuilder,
    TrainingCommand,
    load_training_corpus,
)
from screen2action.training.model_factory import build_screen2action_model
from screen2action.training.quantization import convert_per_channel_ptq, enable_per_channel_qat

ACTION_NAMES = (
    ActionType.CLICK.value,
    ActionType.DRAG.value,
    ActionType.SCROLL.value,
    ActionType.LONG_PRESS.value,
)


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return cast(dict[str, object], value)


def _positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _int_list(value: object, *, context: str) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        values: tuple[int, ...] = (value,)
    elif isinstance(value, list):
        values = tuple(_positive_int(item, context=context) for item in value)
    else:
        raise ValueError(f"{context} must be an integer or list of integers")
    if not values:
        raise ValueError(f"{context} cannot be empty")
    return tuple(sorted(set(values)))


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Resolved single-run evaluation policy."""

    source: str
    split: str
    batch_size: int
    recall_ks: tuple[int, ...]
    top_k: int
    ssb_budget: int
    proposal_iou_threshold: float
    selector_variant: str
    graph_relations: str
    relation_reranking: bool
    visual_variant: str
    quantization_variant: str
    max_commands: int | None

    @classmethod
    def from_config(
        cls,
        config: ResolvedConfig,
        *,
        split: str | None = None,
        max_commands: int | None = None,
    ) -> EvaluationSettings:
        values = cast(Mapping[str, object], config.values)
        evaluation = _mapping(values.get("evaluation"), context="evaluation")
        model = _mapping(values.get("model"), context="model")
        source = str(evaluation.get("source", "all"))
        selected_split = split or str(evaluation.get("split", "test"))
        if selected_split not in {"train", "val", "test"}:
            raise ValueError("evaluation split must be train, val, or test")
        raw_ks = evaluation.get(
            "retrieval_top_k",
            values.get("retrieval_top_k", model.get("top_k", 8)),
        )
        configured_ks = _int_list(raw_ks, context="evaluation.retrieval_top_k")
        recall_ks = tuple(sorted(set((1, 4, 8, *configured_ks))))
        top_k = max(configured_ks)
        raw_budgets = evaluation.get(
            "budgets",
            evaluation.get("ssb_budget", values.get("ssb_budgets", model.get("ssb_budget", 512))),
        )
        budgets = _int_list(raw_budgets, context="evaluation.budgets")
        iou_threshold = float(str(evaluation.get("proposal_iou_threshold", 0.5)))
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("evaluation.proposal_iou_threshold must be in (0, 1]")
        batch_size = _positive_int(
            evaluation.get("batch_size", 1),
            context="evaluation.batch_size",
        )
        selector = str(evaluation.get("selector_variant", "learned"))
        graph_relations = str(evaluation.get("graph_relations", "all"))
        quantization = str(evaluation.get("quantization_variant", "fp32"))
        visual_variant = str(evaluation.get("visual_variant", "mobilevit_accurate"))
        if selector not in {"learned", "confidence_only"}:
            raise ValueError("evaluation selector must be learned or confidence_only")
        if graph_relations not in {"all", "none", "no_ordinal", "no_proximity"}:
            raise ValueError("evaluation graph_relations value is unsupported")
        if quantization not in {"fp32", "ptq_weight_only", "qat_fake_quant"}:
            raise ValueError("evaluation quantization variant is unsupported")
        if visual_variant != "mobilevit_accurate":
            raise ValueError(
                "only mobilevit_accurate is implemented; ROIAlign Fast is optional and unsupported"
            )
        if max_commands is not None and max_commands <= 0:
            raise ValueError("max_commands must be positive")
        return cls(
            source=source,
            split=selected_split,
            batch_size=batch_size,
            recall_ks=recall_ks,
            top_k=top_k,
            ssb_budget=max(budgets),
            proposal_iou_threshold=iou_threshold,
            selector_variant=selector,
            graph_relations=graph_relations,
            relation_reranking=bool(evaluation.get("relation_reranking", True)),
            visual_variant=visual_variant,
            quantization_variant=quantization,
            max_commands=max_commands,
        )


@dataclass(frozen=True, slots=True)
class CalibrationBinding:
    temperature: float
    threshold: float | None
    checkpoint_sha256: str
    validation_manifest_digest: str


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    status: str
    artifact_set: EvaluationArtifactSet
    metrics: Mapping[str, object]
    settings: EvaluationSettings

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "artifacts": asdict(self.artifact_set),
            "metrics": dict(self.metrics),
            "settings": asdict(self.settings),
        }


def _derived_config(config: ResolvedConfig, settings: EvaluationSettings) -> ResolvedConfig:
    values = deepcopy(dict(config.values))
    model_raw = values.get("model")
    model = dict(model_raw) if isinstance(model_raw, dict) else {}
    model["top_k"] = settings.top_k
    model["ssb_budget"] = settings.ssb_budget
    values["model"] = model
    return ResolvedConfig(
        cast(Any, values),
        source_paths=config.source_paths,
        overrides=(
            *config.overrides,
            f"evaluation.runtime_top_k={settings.top_k}",
            f"evaluation.runtime_ssb_budget={settings.ssb_budget}",
        ),
    )


def _load_calibration(
    path: Path | None,
    *,
    checkpoint_digest: str,
) -> CalibrationBinding | None:
    if path is None:
        return None
    temperature_path = path / "temperature.json" if path.is_dir() else path
    threshold_path = temperature_path.parent / "threshold-policy.json"
    if not temperature_path.is_file():
        raise FileNotFoundError(f"temperature artifact is missing: {temperature_path}")
    raw = json.loads(temperature_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("binding"), dict):
        raise ValueError("temperature artifact is malformed")
    binding = cast(dict[str, object], raw["binding"])
    if str(binding.get("checkpoint_sha256", "")) != checkpoint_digest:
        raise ValueError("calibration artifact checkpoint digest mismatch")
    temperature = float(str(raw.get("temperature", 0.0)))
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("calibration artifact temperature is invalid")
    threshold: float | None = None
    if threshold_path.is_file():
        threshold_raw = json.loads(threshold_path.read_text(encoding="utf-8"))
        if not isinstance(threshold_raw, dict) or not isinstance(threshold_raw.get("policy"), dict):
            raise ValueError("threshold artifact is malformed")
        if threshold_raw.get("binding") != binding:
            raise ValueError("threshold artifact binding does not match temperature artifact")
        if threshold_raw.get("temperature_artifact_sha256") != sha256_file(temperature_path):
            raise ValueError("threshold artifact temperature digest mismatch")
        threshold = float(str(threshold_raw["policy"].get("threshold")))
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0000001:
            raise ValueError("threshold artifact probability is invalid")
    return CalibrationBinding(
        temperature,
        threshold,
        checkpoint_digest,
        str(binding.get("validation_manifest_digest", "")),
    )


def _batches(
    commands: Sequence[TrainingCommand], batch_size: int
) -> tuple[tuple[TrainingCommand, ...], ...]:
    ordered = sorted(commands, key=lambda item: (item.screen_id, item.command_id))
    return tuple(
        tuple(ordered[offset : offset + batch_size])
        for offset in range(0, len(ordered), batch_size)
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(device: torch.device, started: float) -> float:
    _sync(device)
    return (time.perf_counter() - started) * 1000.0


def _box(row: Mapping[str, object]) -> Box:
    return cast(
        Box,
        tuple(float(str(row[f"target_{axis}"])) for axis in ("x1", "y1", "x2", "y2")),
    )


def _selected_ids(output: object, screen_index: int) -> tuple[int, ...]:
    encoded = cast(Any, output)
    return tuple(
        node.node_id
        for index, node in enumerate(encoded.records[screen_index])
        if bool(encoded.selected_node_mask[screen_index, index])
    )


def _predicted_parameters(
    grounded: Any,
    command_index: int,
    selected_rank: int | None,
) -> tuple[float, ...]:
    result = torch.zeros(9, dtype=torch.float32)
    if selected_rank is None:
        return tuple(float(value) for value in result)
    tensors = grounded.tensors
    result[:2] = tensors.scroll_delta[command_index, selected_rank].detach().cpu()
    source_rank = int(tensors.drag_source_logits[command_index].argmax())
    destination_rank = int(tensors.drag_destination_logits[command_index].argmax())
    for rank, output, offset in (
        (source_rank, tensors.drag_source_point_local, 2),
        (destination_rank, tensors.drag_destination_point_local, 4),
    ):
        local = tuple(float(value) for value in output[command_index, rank].detach().cpu())
        box = tuple(
            float(value)
            for value in grounded.expanded_crop_boxes[command_index, rank].detach().cpu()
        )
        point = crop_to_screen_point(cast(Point, local), cast(Box, box))
        result[offset : offset + 2] = torch.tensor(point)
    result[6] = tensors.drag_duration[command_index, source_rank, 0].detach().cpu()
    long_local = tuple(
        float(value)
        for value in tensors.long_press_point_local[command_index, selected_rank].detach().cpu()
    )
    selected_box = tuple(
        float(value)
        for value in grounded.expanded_crop_boxes[command_index, selected_rank].detach().cpu()
    )
    result[7:9] = torch.tensor(
        crop_to_screen_point(cast(Point, long_local), cast(Box, selected_box))
    )
    return tuple(float(value) for value in result)


def _failure_reason(
    *,
    proposal_hit: bool,
    target_survived: bool | None,
    retrieved: bool,
    selected_correct: bool | None,
    point_correct: bool,
    action_correct: bool | None,
    abstained: bool,
) -> str:
    if not proposal_hit:
        return "proposal_miss"
    if target_survived is False:
        return "target_pruned"
    if not retrieved:
        return "target_not_retrieved"
    if selected_correct is False:
        return "wrong_candidate"
    if not point_correct:
        return "point_outside_target"
    if action_correct is False:
        return "wrong_action"
    if abstained:
        return "abstained"
    return "success"


def _peak_process_memory() -> tuple[str, int]:
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if isinstance(peak, int):
            return "process_peak_working_set", peak
        return "process_resident_set_at_completion", int(info.rss)
    except (ImportError, OSError, AttributeError):
        return "unavailable", 0


def run_evaluation(
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
) -> EvaluationRunResult:
    """Evaluate one immutable checkpoint and emit direct per-example evidence."""

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but torch.cuda.is_available() is false")
    torch_device = torch.device(device)
    settings = EvaluationSettings.from_config(
        config,
        split=split,
        max_commands=max_commands,
    )
    resolved = _derived_config(config, settings)
    corpus = load_training_corpus(manifest)
    commands = [
        command
        for command in corpus.split_commands(settings.split)
        if settings.source == "all"
        or str(corpus.screens[command.screen_id]["source_dataset"]) == settings.source
    ]
    if settings.max_commands is not None:
        commands = commands[: settings.max_commands]
    if not commands:
        raise ValueError("evaluation selected no commands from the frozen manifest")
    selected_screens = [corpus.screens[screen_id] for screen_id in {c.screen_id for c in commands}]
    if settings.source == "screenspot":
        require_evaluation_only_screenspot(selected_screens, split=settings.split)
    root = repository_root()
    cache_root = model_cache_root or root / ".cache" / "models"
    loader = CachedPerceptionLoader(
        cache_manifest,
        data_manifest_digest=corpus.manifest_digest,
    )
    bundle = build_screen2action_model(
        cast(Mapping[str, object], resolved.values),
        tokenizer_vocab_size=corpus.tokenizer.vocab_size,
        model_lock_path=model_lock,
        cache_root=cache_root,
    )
    if bundle.profile == "paper_reference" and (
        loader.model_bundle_lock_digest != bundle.model_lock_digest
    ):
        raise ValueError("evaluation cache and trainable model use different model locks")
    if settings.quantization_variant == "qat_fake_quant":
        converted = enable_per_channel_qat(bundle.model)
        if converted <= 0:
            raise ValueError("QAT evaluation found no supported layers")
    cache_digest = manifest_hash(cache_manifest)
    load_model_weights(
        checkpoint,
        bundle.model,
        device=torch_device,
        expected_manifest_hash=corpus.manifest_digest,
        expected_model_lock_hash=bundle.model_lock_digest,
        expected_cache_manifest_hash=cache_digest,
    )
    if settings.quantization_variant == "ptq_weight_only":
        converted = convert_per_channel_ptq(bundle.model)
        if converted <= 0:
            raise ValueError("PTQ evaluation found no supported layers")
    bundle.model.to(torch_device).eval()
    builder = CachedScreenBatchBuilder(corpus, loader, bundle.model)
    checkpoint_digest = sha256_file(checkpoint)
    calibration_binding = _load_calibration(
        calibration,
        checkpoint_digest=checkpoint_digest,
    )
    temperature = calibration_binding.temperature if calibration_binding is not None else 1.0
    threshold = calibration_binding.threshold if calibration_binding is not None else None
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    tracemalloc.start()
    records: list[EvaluationRecord] = []
    try:
        with torch.no_grad():
            for raw_batch in _batches(commands, settings.batch_size):
                started = time.perf_counter()
                batch = builder.build(raw_batch)
                data_ms = _elapsed_ms(torch_device, started)
                started = time.perf_counter()
                encoded = bundle.model.encode_frames(
                    batch.perceived_frames,
                    screenshots=batch.images,
                    stochastic_selector=False,
                    selector_variant=settings.selector_variant,
                    graph_relations=settings.graph_relations,
                )
                graph_ms = _elapsed_ms(torch_device, started)
                started = time.perf_counter()
                grounded = bundle.model.ground_commands(
                    encoded,
                    batch.commands,
                    relation_reranking=settings.relation_reranking,
                    diagnostic_top_k=max(settings.recall_ks),
                )
                grounding_ms = _elapsed_ms(torch_device, started)
                started = time.perf_counter()
                batch_count = len(raw_batch)
                for command_index, command in enumerate(raw_batch):
                    row = command.row
                    screen_index = int(batch.commands.screen_indices[command_index])
                    frame = batch.perceived_frames[screen_index]
                    target_box = _box(row)
                    proposals = tuple(
                        node for node in frame.nodes if node.node_type is not NodeType.ROOT
                    )
                    match = match_positive_proposal(
                        target_box,
                        proposals,
                        iou_threshold=settings.proposal_iou_threshold,
                    )
                    selected_ids = _selected_ids(encoded, screen_index)
                    proposal_hit = match.proposal_id is not None
                    target_survived = (
                        match.proposal_id in selected_ids if match.proposal_id is not None else None
                    )
                    reference_ids = tuple(
                        int(value)
                        for value, valid in zip(
                            batch.commands.reference_node_ids[command_index].detach().cpu(),
                            batch.commands.reference_mask[command_index].detach().cpu(),
                            strict=True,
                        )
                        if bool(valid)
                    )
                    reference_survived = (
                        all(value in selected_ids for value in reference_ids)
                        if reference_ids
                        else None
                    )
                    valid_candidates = grounded.candidate_valid_mask[command_index]
                    candidate_ids = tuple(
                        int(value)
                        for value, valid in zip(
                            grounded.candidate_node_ids[command_index].detach().cpu(),
                            valid_candidates.detach().cpu(),
                            strict=True,
                        )
                        if bool(valid)
                    )
                    candidate_scores = tuple(
                        float(value)
                        for value, valid in zip(
                            grounded.retrieval_scores[command_index].detach().cpu(),
                            valid_candidates.detach().cpu(),
                            strict=True,
                        )
                        if bool(valid)
                    )
                    retrieval_hits = {
                        str(k): bool(
                            match.proposal_id is not None
                            and bool(
                                (
                                    (
                                        grounded.ranked_node_ids[command_index, :k]
                                        == match.proposal_id
                                    )
                                    & grounded.ranked_valid_mask[command_index, :k]
                                ).any()
                            )
                        )
                        for k in settings.recall_ks
                    }
                    selected_rank: int | None = None
                    selected_id: int | None = None
                    if candidate_ids:
                        selected_rank = int(
                            grounded.tensors.candidate_logits[command_index].argmax()
                        )
                        if selected_rank < len(candidate_ids):
                            selected_id = candidate_ids[selected_rank]
                        else:
                            selected_rank = None
                    target_present = bool(
                        match.proposal_id is not None and match.proposal_id in candidate_ids
                    )
                    selected_correct = selected_id == match.proposal_id if target_present else None
                    predicted_point: Point | None = None
                    if selected_rank is not None:
                        local = tuple(
                            float(value)
                            for value in grounded.tensors.point_local[command_index, selected_rank]
                            .detach()
                            .cpu()
                        )
                        crop_box = tuple(
                            float(value)
                            for value in grounded.expanded_crop_boxes[command_index, selected_rank]
                            .detach()
                            .cpu()
                        )
                        predicted_point = crop_to_screen_point(
                            cast(Point, local),
                            cast(Box, crop_box),
                        )
                    point_correct = official_point_in_target(predicted_point, target_box)
                    point_given_candidate = point_correct if selected_correct is True else None
                    predicted_action: str | None = None
                    if selected_rank is not None:
                        action_index = int(
                            grounded.tensors.action_type_logits[
                                command_index, selected_rank
                            ].argmax()
                        )
                        predicted_action = ACTION_NAMES[action_index]
                    action_labeled = bool(batch.commands.action_mask[command_index])
                    target_action = str(row["action_type"])
                    action_correct = predicted_action == target_action if action_labeled else None
                    predicted_parameters = _predicted_parameters(
                        grounded,
                        command_index,
                        selected_rank,
                    )
                    target_parameters = tuple(
                        float(value)
                        for value in batch.commands.parameter_targets[command_index].detach().cpu()
                    )
                    parameter_mask = tuple(
                        bool(value)
                        for value in batch.commands.parameter_mask[command_index].detach().cpu()
                    )
                    parameter_errors = [
                        abs(predicted - target)
                        for predicted, target, valid in zip(
                            predicted_parameters,
                            target_parameters,
                            parameter_mask,
                            strict=True,
                        )
                        if valid
                    ]
                    parameter_error = (
                        sum(parameter_errors) / len(parameter_errors) if parameter_errors else None
                    )
                    confidence_logit: float | None = None
                    confidence_probability: float | None = None
                    if selected_rank is not None:
                        confidence_logit = float(
                            grounded.tensors.confidence_logits[command_index, selected_rank]
                            .detach()
                            .cpu()
                        )
                        confidence_probability = float(
                            torch.sigmoid(torch.tensor(confidence_logit / temperature))
                        )
                    abstained = bool(
                        threshold is not None
                        and (confidence_probability is None or confidence_probability < threshold)
                    )
                    selection = encoded.selections[screen_index]
                    ssb_length = selection.total_cost if selection is not None else 0
                    selected_set = set(selected_ids)
                    edge_count = sum(
                        edge.src in selected_set and edge.dst in selected_set
                        for edge in encoded.edges[screen_index]
                    )
                    target_area = max(0.0, target_box[2] - target_box[0]) * max(
                        0.0, target_box[3] - target_box[1]
                    )
                    screen = corpus.screens[command.screen_id]
                    records.append(
                        EvaluationRecord(
                            command.command_id,
                            command.screen_id,
                            str(screen["source_dataset"]),
                            settings.split,
                            str(screen["platform"]),
                            target_type_for_screen(
                                corpus.elements.get(command.screen_id, {}),
                                target_box,
                            ),
                            str(row["relation_type"]),
                            target_box,
                            target_area,
                            match.proposal_id,
                            match.iou,
                            match.used_center_fallback,
                            proposal_hit,
                            selected_ids,
                            target_survived,
                            bool(reference_ids),
                            reference_survived,
                            candidate_ids,
                            candidate_scores,
                            retrieval_hits,
                            selected_id,
                            selected_rank,
                            selected_correct,
                            predicted_point,
                            point_given_candidate,
                            point_correct,
                            predicted_action,
                            target_action,
                            action_labeled,
                            action_correct,
                            predicted_parameters,
                            target_parameters,
                            parameter_mask,
                            parameter_error,
                            confidence_logit,
                            confidence_probability,
                            calibration_binding is not None,
                            abstained,
                            ssb_length,
                            len(selected_ids),
                            edge_count,
                            data_ms / batch_count,
                            graph_ms / batch_count,
                            grounding_ms / batch_count,
                            0.0,
                            _failure_reason(
                                proposal_hit=proposal_hit,
                                target_survived=target_survived,
                                retrieved=retrieval_hits[str(settings.top_k)],
                                selected_correct=selected_correct,
                                point_correct=point_correct,
                                action_correct=action_correct,
                                abstained=abstained,
                            ),
                        )
                    )
                postprocess_ms = _elapsed_ms(torch_device, started)
                count = len(raw_batch)
                for offset in range(len(records) - count, len(records)):
                    records[offset] = replace(
                        records[offset],
                        latency_postprocess_ms=postprocess_ms / count,
                    )
        _, python_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    memory_kind, process_memory = _peak_process_memory()
    metrics = aggregate_evaluation_records(
        records,
        recall_ks=settings.recall_ks,
        confidence_temperature=temperature,
    )
    metrics["memory"] = {
        "process_measurement": memory_kind,
        "process_bytes": process_memory,
        "python_tracemalloc_peak_bytes": int(python_peak),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(torch_device)) if torch_device.type == "cuda" else 0
        ),
        "measured_host": True,
        "mobile_runtime_measured": False,
    }
    policy = {
        **asdict(settings),
        "point_rule": "inclusive_normalized_point_in_target",
        "latency_scope": "observed_batch_stage_time_repeated_per_command",
        "calibration_validation_manifest_digest": (
            calibration_binding.validation_manifest_digest
            if calibration_binding is not None
            else None
        ),
    }
    artifacts = write_evaluation_artifacts(
        output_directory,
        records=records,
        metrics=metrics,
        config=resolved,
        checkpoint=checkpoint,
        data_manifest_digest=corpus.manifest_digest,
        cache_manifest_digest=cache_digest,
        root=root,
        device=device,
        evaluation_policy=policy,
    )
    return EvaluationRunResult("complete", artifacts, metrics, settings)
