"""Configuration-driven CPU/CUDA/DDP stage runner with exact resume."""

from __future__ import annotations

import gc
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from screen2action.config import ResolvedConfig, write_resolved_config
from screen2action.handoff import repository_root
from screen2action.models.screen2action_model import Screen2ActionModelOutput
from screen2action.perception.loader import CachedPerceptionLoader
from screen2action.training.artifacts import RunArtifactWriter, write_run_manifest
from screen2action.training.checkpoints import (
    configuration_hash,
    load_checkpoint,
    load_model_weights,
    manifest_hash,
    save_checkpoint,
)
from screen2action.training.config import StageRunSettings
from screen2action.training.data import (
    CachedScreenBatchBuilder,
    TrainingCommand,
    load_training_corpus,
)
from screen2action.training.detector_native import NativeStageResult, run_detector_stage
from screen2action.training.distributed import (
    DistributedContext,
    initialize_distributed,
    wrap_ddp,
)
from screen2action.training.engine import LossResult, Trainer, TrainerConfig, select_precision
from screen2action.training.joint import (
    JointStage3BatchBuilder,
    JointStage3Model,
    JointStage3Output,
)
from screen2action.training.model_factory import TrainModelBundle, build_screen2action_model
from screen2action.training.ocr_native import (
    OcrBatchBuilder,
    OcrExample,
    OcrModelBundle,
    build_locked_ocr_model,
    build_ocr_corpus,
)
from screen2action.training.optimizer import accumulation_plan, build_adamw, build_scheduler
from screen2action.training.quantization import (
    ActivationCalibrator,
    convert_per_channel_ptq,
    enable_per_channel_qat,
    quantization_inventory,
    set_qat_fake_quant,
)
from screen2action.training.seed import capture_rng_state, seed_everything
from screen2action.training.semantics import (
    SemanticBatchBuilder,
    SemanticScreenExample,
    SemanticsGraphOutput,
    SemanticsModelBundle,
    build_semantic_corpus,
    build_semantics_graph_model,
)
from screen2action.training.stages import configure_stage_trainability


@dataclass(frozen=True, slots=True)
class StageRunResult:
    """Stable summary emitted by CLI and tests after a stage invocation."""

    status: str
    stage: str
    run_directory: str
    rank: int
    world_size: int
    device: str
    precision: str
    epochs_completed: int
    global_step: int
    optimizer_step: int
    best_validation_loss: float | None
    latest_checkpoint: str | None
    best_checkpoint: str | None
    effective_global_batch: int
    accumulation_steps: int
    trainable_parameters: int
    frozen_parameters: int
    data_manifest_digest: str
    model_lock_digest: str
    cache_manifest_sha256: str
    run_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class BatchProbeResult:
    """Largest empirically successful command/screen batch on this device."""

    stage: str
    device: str
    precision: str
    world_size: int
    largest_successful_per_device_batch: int
    peak_vram_bytes: int
    recommended_accumulation_for_global_256: int
    effective_global_batch: int
    attempted_batches: tuple[int, ...]
    first_oom_batch: int | None
    oom_message: str | None


@dataclass(frozen=True, slots=True)
class _StageRuntime:
    model: nn.Module
    raw_model: nn.Module
    loss: Callable[[nn.Module, object, int], LossResult]
    train_batches: Callable[[int], Sequence[object]]
    validation_batches: Callable[[int], Sequence[object]]
    model_lock_digest: str
    cache_manifest_sha256: str
    pretrained_prefixes: tuple[str, ...]
    trainability: Mapping[str, int]
    set_epoch: Callable[[int], None]


def _screen_loss(
    builder: CachedScreenBatchBuilder,
    *,
    stage_name: str,
    epoch_state: list[int],
    settings: StageRunSettings,
) -> Callable[[nn.Module, object, int], LossResult]:
    model_stage = (
        "stage2_grounding" if stage_name == "stage2_selector_retrieval" else "stage3_joint"
    )

    def compute(module: nn.Module, raw_batch: object, step: int) -> LossResult:
        commands = cast(Sequence[TrainingCommand], raw_batch)
        progress = min(epoch_state[0] / max(settings.epochs, 1), 1.0)
        anneal = min(progress / max(settings.gumbel_anneal_fraction, 1e-9), 1.0)
        temperature = settings.gumbel_temperature_start + anneal * (
            settings.gumbel_temperature_end - settings.gumbel_temperature_start
        )
        forced = (
            settings.forced_positive_probability
            if stage_name == "stage2_selector_retrieval" and epoch_state[0] < 2
            else 0.0
        )
        data_started = time.perf_counter()
        batch = builder.build(commands)
        data_time = time.perf_counter() - data_started
        output = cast(
            Screen2ActionModelOutput,
            module(
                batch,
                stage=model_stage,
                step=step,
                selector_temperature=temperature,
                forced_positive_probability=forced,
            ),
        )
        grounding = output.grounding_loss
        retention = output.retention_loss
        valid_candidates = output.grounded_commands.candidate_valid_mask.sum()
        supervised = output.loss_inputs.grounding.candidate_mask.sum()
        hard_negatives = (valid_candidates - supervised).clamp_min(0)
        candidate_mask = output.loss_inputs.grounding.candidate_mask
        candidate_accuracy = (
            (
                output.grounded_commands.tensors.candidate_logits.argmax(dim=-1)[candidate_mask]
                == output.loss_inputs.grounding.candidate_indices[candidate_mask]
            )
            .float()
            .mean()
            if bool(candidate_mask.any())
            else output.total_loss.detach() * 0.0
        )
        components: dict[str, torch.Tensor | float] = {
            "grounding_candidate_loss": grounding.candidate,
            "grounding_point_loss": grounding.point,
            "grounding_action_loss": grounding.action,
            "grounding_parameter_loss": grounding.parameters,
            "ui_contrastive_loss": output.grounded_commands.retrieval_loss,
            "confidence_loss": grounding.confidence,
            "target_survival_loss": retention.target_survival,
            "reference_survival_loss": retention.reference_survival,
            "budget_loss": retention.budget,
            "gumbel_temperature": temperature,
            "forced_positive_probability": forced,
            "forced_positive_insertions": output.grounded_commands.forced_positive_mask.sum(),
            "same_screen_hard_negatives": hard_negatives,
            "grounding_candidate_accuracy": candidate_accuracy,
        }
        return LossResult(output.total_loss, components, len(commands), data_time)

    return compute


def _semantic_loss(
    builder: SemanticBatchBuilder,
) -> Callable[[nn.Module, object, int], LossResult]:
    def compute(module: nn.Module, raw_batch: object, step: int) -> LossResult:
        del step
        examples = cast(Sequence[SemanticScreenExample], raw_batch)
        data_started = time.perf_counter()
        batch = builder.build(examples)
        data_time = time.perf_counter() - data_started
        output = cast(SemanticsGraphOutput, module(batch))
        return LossResult(
            output.total_loss,
            {
                "perception_icon_loss": output.direct_icon_loss,
                "perception_actionability_loss": output.direct_actionability_loss,
                "graph_icon_loss": output.graph_icon_loss,
                "graph_actionability_loss": output.graph_actionability_loss,
                "icon_supervision_count": float(output.icon_count),
                "actionability_supervision_count": float(output.actionability_count),
            },
            len(examples),
            data_time,
        )

    return compute


def _joint_loss(
    builder: JointStage3BatchBuilder,
    *,
    epoch_state: list[int],
    settings: StageRunSettings,
) -> Callable[[nn.Module, object, int], LossResult]:
    def compute(module: nn.Module, raw_batch: object, step: int) -> LossResult:
        commands = cast(Sequence[TrainingCommand], raw_batch)
        progress = min(epoch_state[0] / max(settings.epochs, 1), 1.0)
        anneal = min(progress / max(settings.gumbel_anneal_fraction, 1e-9), 1.0)
        temperature = settings.gumbel_temperature_start + anneal * (
            settings.gumbel_temperature_end - settings.gumbel_temperature_start
        )
        data_started = time.perf_counter()
        batch = builder.build(commands)
        data_time = time.perf_counter() - data_started
        output = cast(
            JointStage3Output,
            module(
                batch,
                step=step,
                selector_temperature=temperature,
            ),
        )
        downstream = output.downstream
        grounding = downstream.grounding_loss
        retention = downstream.retention_loss
        candidate_mask = downstream.loss_inputs.grounding.candidate_mask
        candidate_accuracy = (
            (
                downstream.grounded_commands.tensors.candidate_logits.argmax(dim=-1)[candidate_mask]
                == downstream.loss_inputs.grounding.candidate_indices[candidate_mask]
            )
            .float()
            .mean()
            if bool(candidate_mask.any())
            else output.total_loss.detach() * 0.0
        )
        components: dict[str, torch.Tensor | float] = {
            "grounding_candidate_loss": grounding.candidate,
            "grounding_point_loss": grounding.point,
            "grounding_action_loss": grounding.action,
            "grounding_parameter_loss": grounding.parameters,
            "ui_contrastive_loss": downstream.grounded_commands.retrieval_loss,
            "confidence_loss": grounding.confidence,
            "target_survival_loss": retention.target_survival,
            "reference_survival_loss": retention.reference_survival,
            "budget_loss": retention.budget,
            "stage3_semantic_perception_loss": output.semantic_loss,
            "gumbel_temperature": temperature,
            "same_screen_hard_negatives": (
                downstream.grounded_commands.candidate_valid_mask.sum()
                - downstream.loss_inputs.grounding.candidate_mask.sum()
            ).clamp_min(0),
            "grounding_candidate_accuracy": candidate_accuracy,
        }
        if output.semantics is not None:
            components.update(
                {
                    "perception_icon_loss": output.semantics.direct_icon_loss,
                    "perception_actionability_loss": output.semantics.direct_actionability_loss,
                    "graph_icon_loss": output.semantics.graph_icon_loss,
                    "graph_actionability_loss": output.semantics.graph_actionability_loss,
                }
            )
        else:
            zero = output.semantic_loss.detach()
            components.update(
                {
                    "perception_icon_loss": zero,
                    "perception_actionability_loss": zero,
                    "graph_icon_loss": zero,
                    "graph_actionability_loss": zero,
                }
            )
        return LossResult(output.total_loss, components, len(commands), data_time)

    return compute


def _ocr_loss(
    builder: OcrBatchBuilder,
) -> Callable[[nn.Module, object, int], LossResult]:
    def compute(module: nn.Module, raw_batch: object, step: int) -> LossResult:
        del step
        examples = cast(Sequence[OcrExample], raw_batch)
        data_started = time.perf_counter()
        batch = builder.build(examples)
        data_time = time.perf_counter() - data_started
        loss = cast(torch.Tensor, module(batch))
        return LossResult(
            loss,
            {"ocr_ctc_loss": loss.detach()},
            len(examples),
            data_time,
        )

    return compute


def _command_runtime(
    config: ResolvedConfig,
    settings: StageRunSettings,
    *,
    context: DistributedContext,
    manifest: Path,
    cache_manifest: Path | None,
    model_lock: Path | None,
    model_cache_root: Path,
) -> _StageRuntime:
    if cache_manifest is None:
        raise ValueError(f"{settings.stage} requires --cache-manifest")
    corpus = load_training_corpus(manifest)
    loader = CachedPerceptionLoader(
        cache_manifest,
        data_manifest_digest=corpus.manifest_digest,
    )
    bundle: TrainModelBundle = build_screen2action_model(
        cast(Mapping[str, object], config.values),
        tokenizer_vocab_size=corpus.tokenizer.vocab_size,
        model_lock_path=model_lock,
        cache_root=model_cache_root,
    )
    downstream_trainability = configure_stage_trainability(bundle.model, settings.stage)
    if settings.stage == "stage4_qat":
        converted = enable_per_channel_qat(bundle.model)
        if converted <= 0:
            raise ValueError("stage4_qat found no supported Linear or Conv2d layers")
        downstream_trainability = {
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in bundle.model.parameters()
                if parameter.requires_grad
            ),
            "frozen_parameters": sum(
                parameter.numel()
                for parameter in bundle.model.parameters()
                if not parameter.requires_grad
            ),
        }
    bundle.model.to(context.device)
    builder = CachedScreenBatchBuilder(corpus, loader, bundle.model)
    epoch_state = [0]
    crop_frozen: list[bool | None] = [None]

    def set_epoch(epoch: int) -> None:
        epoch_state[0] = epoch
        frozen = (
            settings.stage == "stage2_selector_retrieval"
            and epoch < settings.frozen_crop_warmup_epochs
        )
        if crop_frozen[0] != frozen:
            bundle.model.set_crop_encoder_updates_frozen(frozen)
            crop_frozen[0] = frozen

    def batches(split: str, epoch: int, training: bool) -> Sequence[object]:
        if not corpus.split_commands(split):
            if split == "val" and not settings.validation_required:
                return ()
            raise ValueError(f"frozen manifest has no commands in required split {split!r}")
        return corpus.batches(
            split,
            epoch=epoch,
            seed=settings.seed,
            rank=context.rank,
            world_size=context.world_size,
            per_device_batch_size=settings.per_device_batch_size,
            training=training,
            max_commands=settings.max_commands,
        )

    if settings.stage == "stage3_joint":
        semantic_corpus = build_semantic_corpus(corpus)
        semantic_bundle = build_semantics_graph_model(
            cast(Mapping[str, object], config.values),
            model_lock_path=model_lock,
            cache_root=model_cache_root,
        )
        if semantic_bundle.model_lock_digest != bundle.model_lock_digest:
            raise ValueError("stage3 downstream and semantic branches use different model locks")
        semantic_by_screen = {
            example.screen_id: example
            for values in semantic_corpus.by_split.values()
            for example in values
        }
        semantic_builder = SemanticBatchBuilder(
            device=context.device,
            crop_size=semantic_bundle.crop_size,
        )
        joint_raw = config.values.get("joint_native_branches")
        joint_values = cast(dict[str, object], joint_raw) if isinstance(joint_raw, dict) else {}
        semantic_weight = float(str(joint_values.get("semantic_perception_weight", 1.0)))
        joint = JointStage3Model(
            bundle.model,
            semantic_bundle.model,
            semantic_weight=semantic_weight,
        ).to(context.device)
        joint_builder = JointStage3BatchBuilder(builder, semantic_builder, semantic_by_screen)
        trainability = {
            "trainable_parameters": sum(
                parameter.numel() for parameter in joint.parameters() if parameter.requires_grad
            ),
            "frozen_parameters": sum(
                parameter.numel() for parameter in joint.parameters() if not parameter.requires_grad
            ),
        }
        wrapped = wrap_ddp(joint, context, find_unused_parameters=True)
        return _StageRuntime(
            wrapped,
            joint,
            _joint_loss(joint_builder, epoch_state=epoch_state, settings=settings),
            lambda epoch: batches("train", epoch, True),
            lambda epoch: batches("val", epoch, False),
            bundle.model_lock_digest,
            manifest_hash(cache_manifest),
            tuple(f"downstream.{prefix}" for prefix in bundle.pretrained_prefixes)
            + tuple(f"semantics.{prefix}" for prefix in semantic_bundle.pretrained_prefixes),
            trainability,
            set_epoch,
        )
    trainability = downstream_trainability
    wrapped = wrap_ddp(bundle.model, context, find_unused_parameters=True)
    return _StageRuntime(
        wrapped,
        bundle.model,
        _screen_loss(
            builder,
            stage_name=settings.stage,
            epoch_state=epoch_state,
            settings=settings,
        ),
        lambda epoch: batches("train", epoch, True),
        lambda epoch: batches("val", epoch, False),
        bundle.model_lock_digest,
        manifest_hash(cache_manifest),
        bundle.pretrained_prefixes,
        trainability,
        set_epoch,
    )


def _semantics_runtime(
    config: ResolvedConfig,
    settings: StageRunSettings,
    *,
    context: DistributedContext,
    manifest: Path,
    model_lock: Path | None,
    model_cache_root: Path,
) -> _StageRuntime:
    corpus = load_training_corpus(manifest)
    semantic = build_semantic_corpus(corpus)
    bundle: SemanticsModelBundle = build_semantics_graph_model(
        cast(Mapping[str, object], config.values),
        model_lock_path=model_lock,
        cache_root=model_cache_root,
    )
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(True)
    trainability = {
        "trainable_parameters": sum(parameter.numel() for parameter in bundle.model.parameters()),
        "frozen_parameters": 0,
    }
    bundle.model.to(context.device)
    builder = SemanticBatchBuilder(device=context.device, crop_size=bundle.crop_size)

    def batches(split: str, epoch: int, training: bool) -> Sequence[object]:
        if not semantic.by_split.get(split):
            if split == "val" and not settings.validation_required:
                return ()
            raise ValueError(f"frozen manifest has no labeled semantic screens in split {split!r}")
        return semantic.batches(
            split,
            epoch=epoch,
            seed=settings.seed,
            rank=context.rank,
            world_size=context.world_size,
            per_device_batch_size=settings.per_device_batch_size,
            training=training,
            max_screens=settings.max_commands,
        )

    wrapped = wrap_ddp(bundle.model, context, find_unused_parameters=True)
    return _StageRuntime(
        wrapped,
        bundle.model,
        _semantic_loss(builder),
        lambda epoch: batches("train", epoch, True),
        lambda epoch: batches("val", epoch, False),
        bundle.model_lock_digest,
        "not-applicable-stage1-source-crops",
        bundle.pretrained_prefixes,
        trainability,
        lambda epoch: None,
    )


def _ocr_runtime(
    config: ResolvedConfig,
    settings: StageRunSettings,
    *,
    context: DistributedContext,
    manifest: Path,
    model_lock: Path | None,
    model_cache_root: Path,
) -> _StageRuntime:
    corpus = load_training_corpus(manifest)
    ocr_corpus = build_ocr_corpus(corpus)
    bundle: OcrModelBundle = build_locked_ocr_model(
        cast(Mapping[str, object], config.values),
        model_lock_path=model_lock,
        cache_root=model_cache_root,
    )
    bundle.model.to(context.device)
    builder = OcrBatchBuilder(
        context.device,
        height=bundle.crop_height,
        width=bundle.crop_width,
    )

    def batches(split: str, epoch: int, training: bool) -> Sequence[object]:
        if not ocr_corpus.by_split.get(split):
            if split == "val" and not settings.validation_required:
                return ()
            raise ValueError(f"stage1_ocr has no required text crops in split {split!r}")
        return ocr_corpus.batches(
            split,
            epoch=epoch,
            seed=settings.seed,
            rank=context.rank,
            world_size=context.world_size,
            per_device_batch_size=settings.per_device_batch_size,
            training=training,
            max_samples=settings.max_commands,
        )

    trainability = {
        "trainable_parameters": sum(
            parameter.numel() for parameter in bundle.model.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in bundle.model.parameters()
            if not parameter.requires_grad
        ),
    }
    wrapped = wrap_ddp(bundle.model, context, find_unused_parameters=False)
    return _StageRuntime(
        wrapped,
        bundle.model,
        _ocr_loss(builder),
        lambda epoch: batches("train", epoch, True),
        lambda epoch: batches("val", epoch, False),
        bundle.model_lock_digest,
        "not-applicable-stage1-source-crops",
        bundle.pretrained_prefixes,
        trainability,
        lambda epoch: None,
    )


def _evaluate_quantization_variant(
    model: nn.Module,
    batches: Sequence[object],
    loss_function: Callable[[nn.Module, object, int], LossResult],
    *,
    context: DistributedContext,
    step: int,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    examples = 0
    accuracy_sum = 0.0
    with torch.no_grad():
        for batch in batches:
            result = loss_function(model, batch, step)
            loss_sum += float(result.loss.detach().cpu()) * result.examples
            accuracy = result.components.get("grounding_candidate_accuracy", 0.0)
            accuracy_value = (
                float(accuracy.detach().cpu())
                if isinstance(accuracy, torch.Tensor)
                else float(accuracy)
            )
            accuracy_sum += accuracy_value * result.examples
            examples += result.examples
    totals = context.reduce_sum(
        torch.tensor(
            [loss_sum, accuracy_sum, float(examples)],
            dtype=torch.float64,
            device=context.device,
        )
    )
    count = int(totals[2])
    if count <= 0:
        raise ValueError("quantization comparison has no validation examples")
    return {
        "mean_loss": float(totals[0] / count),
        "grounding_candidate_accuracy": float(totals[1] / count),
        "examples": float(count),
    }


def _default_run_directory(
    root: Path,
    settings: StageRunSettings,
    context: DistributedContext,
) -> Path:
    name: object | None = None
    if context.is_main:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"{settings.stage}-{timestamp}"
    resolved = str(context.broadcast_object(name))
    return root / "outputs" / "runs" / resolved


def _build_runtime(
    config: ResolvedConfig,
    settings: StageRunSettings,
    *,
    context: DistributedContext,
    manifest: Path,
    cache_manifest: Path | None,
    model_lock: Path | None,
    model_cache_root: Path,
) -> _StageRuntime:
    if settings.stage == "stage1_semantics_graph":
        return _semantics_runtime(
            config,
            settings,
            context=context,
            manifest=manifest,
            model_lock=model_lock,
            model_cache_root=model_cache_root,
        )
    if settings.stage == "stage1_ocr":
        return _ocr_runtime(
            config,
            settings,
            context=context,
            manifest=manifest,
            model_lock=model_lock,
            model_cache_root=model_cache_root,
        )
    if settings.stage in {"stage2_selector_retrieval", "stage3_joint", "stage4_qat"}:
        return _command_runtime(
            config,
            settings,
            context=context,
            manifest=manifest,
            cache_manifest=cache_manifest,
            model_lock=model_lock,
            model_cache_root=model_cache_root,
        )
    raise ValueError(
        f"{settings.stage} uses its dedicated native, orchestration, or quantization runner"
    )


def run_training_stage(
    config: ResolvedConfig,
    *,
    stage: str,
    manifest: Path,
    cache_manifest: Path | None = None,
    model_lock: Path | None = None,
    model_cache_root: Path | None = None,
    device: str = "cpu",
    run_directory: Path | None = None,
    resume: Path | None = None,
    init_checkpoint: Path | None = None,
    max_commands: int | None = None,
    stop_after_optimizer_steps: int | None = None,
) -> StageRunResult | NativeStageResult:
    """Run or exactly resume one supported stage under torchrun or one process."""

    if stop_after_optimizer_steps is not None and stop_after_optimizer_steps <= 0:
        raise ValueError("stop_after_optimizer_steps must be positive")
    if resume is not None and init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    settings = StageRunSettings.from_values(
        cast(Mapping[str, object], config.values),
        stage_name=stage,
        max_commands=max_commands,
    )
    joint_raw = config.values.get("joint_native_branches")
    joint_values = cast(dict[str, object], joint_raw) if isinstance(joint_raw, dict) else {}
    if settings.stage == "stage3_joint" and bool(
        joint_values.get("detector_native_enabled", False)
    ):
        if resume is not None:
            raise ValueError(
                "resume the downstream Stage-3 lineage with detector_native_enabled=false; "
                "the native detector owns a separate checkpoint lineage"
            )
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError(
                "Stage-3 native detector alternation delegates distribution to Ultralytics; "
                "launch the orchestrator as one process"
            )
        policy = str(joint_values.get("detector_update_policy", "alternating"))
        if policy != "alternating":
            raise ValueError("Stage-3 detector update policy currently supports alternating")
        root = repository_root()
        parent = (
            run_directory
            or root
            / "outputs"
            / "runs"
            / f"stage3_joint-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        ).resolve()
        detector = run_detector_stage(
            config,
            manifest=manifest,
            model_lock=model_lock,
            model_cache_root=(model_cache_root or root / ".cache" / "models"),
            device=device,
            run_directory=parent / "detector-native",
        )
        downstream_values = deepcopy(dict(config.values))
        downstream_joint = downstream_values.get("joint_native_branches")
        if not isinstance(downstream_joint, dict):
            downstream_joint = {}
            downstream_values["joint_native_branches"] = downstream_joint
        downstream_joint["detector_native_enabled"] = False
        downstream_config = ResolvedConfig(
            cast(Mapping[str, Any], downstream_values),
            config.source_paths,
            (*config.overrides, "joint_native_branches.detector_native_enabled=false"),
        )
        downstream = run_training_stage(
            downstream_config,
            stage="stage3_joint",
            manifest=manifest,
            cache_manifest=cache_manifest,
            model_lock=model_lock,
            model_cache_root=model_cache_root,
            device=device,
            run_directory=parent / "downstream-and-semantics",
            max_commands=max_commands,
            stop_after_optimizer_steps=stop_after_optimizer_steps,
        )
        corpus_digest = load_training_corpus(manifest).manifest_digest
        write_resolved_config(config, parent)
        _, parent_digest, _ = write_run_manifest(
            parent,
            root=root,
            stage="stage3_joint",
            config_sha256=config.sha256,
            data_manifest_digest=corpus_digest,
            model_lock_digest=downstream.model_lock_digest,
            cache_manifest_sha256=(
                manifest_hash(cache_manifest)
                if cache_manifest is not None
                else "missing-cache-manifest"
            ),
            world_size=1,
            device_type=device,
        )
        report = {
            "schema_version": 1,
            "update_policy": "alternating",
            "detector_native": asdict(detector),
            "downstream_and_semantics": asdict(downstream),
            "downstream_gradient_through_nms": False,
        }
        (parent / "stage3-alternating-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return NativeStageResult(
            "complete" if downstream.status == detector.status == "complete" else "stopped",
            "stage3_joint",
            parent.as_posix(),
            0,
            1,
            device,
            downstream.epochs_completed,
            downstream.latest_checkpoint,
            downstream.best_checkpoint,
            corpus_digest,
            downstream.model_lock_digest,
            parent_digest,
            "alternating_multi_optimizer",
            report,
        )
    if settings.stage == "stage1_full":
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("stage1_full orchestration must be launched as one process")
        raw_full = config.values.get("stage1_full")
        full = cast(dict[str, object], raw_full) if isinstance(raw_full, dict) else {}
        enabled = {
            "stage1_semantics_graph": bool(full.get("semantics_graph", True)),
            "stage1_detector": bool(full.get("detector", False)),
            "stage1_ocr": bool(full.get("ocr", False)),
        }
        selected = tuple(name for name, active in enabled.items() if active)
        if not selected:
            raise ValueError("stage1_full must enable at least one perception sub-stage")
        root = repository_root()
        parent = (
            run_directory
            or root
            / "outputs"
            / "runs"
            / f"stage1_full-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        ).resolve()
        children: list[StageRunResult | NativeStageResult] = []
        for child_stage in selected:
            children.append(
                run_training_stage(
                    config,
                    stage=child_stage,
                    manifest=manifest,
                    cache_manifest=cache_manifest,
                    model_lock=model_lock,
                    model_cache_root=model_cache_root,
                    device=device,
                    run_directory=parent / child_stage,
                    max_commands=max_commands,
                    stop_after_optimizer_steps=stop_after_optimizer_steps,
                )
            )
        corpus_digest = load_training_corpus(manifest).manifest_digest
        write_resolved_config(config, parent)
        _, parent_digest, _ = write_run_manifest(
            parent,
            root=root,
            stage="stage1_full",
            config_sha256=config.sha256,
            data_manifest_digest=corpus_digest,
            model_lock_digest=children[0].model_lock_digest,
            cache_manifest_sha256="not-applicable-stage1-orchestration",
            world_size=1,
            device_type=device,
        )
        report = {
            "schema_version": 1,
            "stage": "stage1_full",
            "separate_checkpoints": True,
            "sub_stages": [asdict(child) for child in children],
            "downstream_gradient_through_nms": False,
            "downstream_gradient_through_ocr_decoding": False,
        }
        (parent / "stage1-full-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return NativeStageResult(
            "complete" if all(child.status == "complete" for child in children) else "stopped",
            "stage1_full",
            parent.as_posix(),
            0,
            1,
            device,
            max(child.epochs_completed for child in children),
            children[-1].latest_checkpoint,
            children[-1].best_checkpoint,
            corpus_digest,
            children[0].model_lock_digest,
            parent_digest,
            "orchestration",
            report,
        )
    if settings.stage == "stage1_detector":
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError(
                "stage1_detector delegates distribution to Ultralytics; launch it as one process"
            )
        root = repository_root()
        native_run = run_directory or (
            root
            / "outputs"
            / "runs"
            / f"stage1_detector-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        return run_detector_stage(
            config,
            manifest=manifest,
            model_lock=model_lock,
            model_cache_root=(model_cache_root or root / ".cache" / "models"),
            device=device,
            run_directory=native_run.resolve(),
            resume=resume,
        )
    context = initialize_distributed(device)
    writer: RunArtifactWriter | None = None
    try:
        seed_everything(settings.seed + context.rank)
        root = repository_root()
        run_dir = (
            run_directory.resolve()
            if run_directory is not None
            else _default_run_directory(root, settings, context)
        )
        if context.is_main:
            write_resolved_config(config, run_dir)
        context.barrier()
        runtime = _build_runtime(
            config,
            settings,
            context=context,
            manifest=manifest,
            cache_manifest=cache_manifest,
            model_lock=model_lock,
            model_cache_root=(model_cache_root or root / ".cache" / "models"),
        )
        corpus_digest = load_training_corpus(manifest).manifest_digest
        initialization: Mapping[str, Any] | None = None
        if settings.stage == "stage4_qat" and resume is None and init_checkpoint is None:
            raise ValueError("stage4_qat requires --init-checkpoint or --resume")
        if init_checkpoint is not None:
            initialization = load_model_weights(
                init_checkpoint,
                runtime.model,
                device=context.device,
                expected_manifest_hash=corpus_digest,
                expected_model_lock_hash=runtime.model_lock_digest,
                expected_cache_manifest_hash=runtime.cache_manifest_sha256,
            )
        manifest_values: object | None = None
        if context.is_main:
            manifest_values = write_run_manifest(
                run_dir,
                root=root,
                stage=settings.stage,
                config_sha256=config.sha256,
                data_manifest_digest=corpus_digest,
                model_lock_digest=runtime.model_lock_digest,
                cache_manifest_sha256=runtime.cache_manifest_sha256,
                world_size=context.world_size,
                device_type=context.device.type,
                initial_checkpoint_sha256=(
                    None
                    if resume is not None
                    else (manifest_hash(init_checkpoint) if init_checkpoint is not None else "")
                ),
            )
        run_manifest_path, run_manifest_digest, git_revision = cast(
            tuple[Path, str, str],
            context.broadcast_object(manifest_values),
        )
        context.barrier()
        if not run_manifest_path.is_file():
            raise FileNotFoundError("rank-zero run manifest was not published")
        plan = accumulation_plan(
            global_batch_size=settings.global_batch_size,
            per_device_batch_size=settings.per_device_batch_size,
            world_size=context.world_size,
        )
        sample_batches = runtime.train_batches(0)
        optimizer_steps_per_epoch = math.ceil(len(sample_batches) / plan.accumulation_steps)
        total_optimizer_steps = optimizer_steps_per_epoch * settings.epochs
        optimizer = build_adamw(
            runtime.raw_model,
            pretrained_prefixes=runtime.pretrained_prefixes,
            new_head_lr=settings.new_head_lr,
            pretrained_lr=settings.pretrained_lr,
            weight_decay=settings.weight_decay,
        )
        scheduler = build_scheduler(
            optimizer,
            total_steps=total_optimizer_steps,
            warmup_fraction=settings.warmup_fraction,
            schedule=settings.scheduler,
        )
        precision = select_precision(context.device, settings.precision)
        trainer = Trainer(
            runtime.model,
            optimizer,
            scheduler=scheduler,
            config=TrainerConfig(
                device=str(context.device),
                gradient_accumulation_steps=plan.accumulation_steps,
                precision=precision,
                grad_clip_norm=settings.gradient_clip_norm,
                log_interval=settings.log_interval,
            ),
            distributed=context,
        )
        start_epoch = 0
        start_batch = 0
        best_metric: float | None = None
        if resume is not None:
            state = load_checkpoint(
                resume,
                runtime.model,
                optimizer,
                scheduler=scheduler,
                scaler=trainer.scaler,
                device=context.device,
                expected_config_hash=configuration_hash(config.values),
                expected_manifest_hash=corpus_digest,
                expected_model_lock_hash=runtime.model_lock_digest,
                expected_cache_manifest_hash=runtime.cache_manifest_sha256,
                expected_run_manifest_hash=run_manifest_digest,
                rank=context.rank,
            )
            start_epoch = state.epoch
            start_batch = state.next_batch_index
            best_metric = state.best_metric
            if state.sampler_state != {"epoch": state.epoch, "seed": settings.seed}:
                raise ValueError("checkpoint sampler state does not match the deterministic corpus")
            trainer.global_step = state.step
            trainer.optimizer_step = state.optimizer_step
        if context.is_main:
            writer = RunArtifactWriter.open(run_dir, tensorboard=settings.tensorboard)
            writer.write(
                {
                    "event": "run_started" if resume is None else "run_resumed",
                    "stage": settings.stage,
                    "rank": context.rank,
                    "world_size": context.world_size,
                    "device": str(context.device),
                    "precision": precision,
                    "accumulation_steps": plan.accumulation_steps,
                    "effective_global_batch": plan.effective_global_batch,
                    "desired_global_batch": plan.desired_global_batch,
                    **runtime.trainability,
                    "initialization": dict(initialization or {}),
                }
            )

        quantization_report: dict[str, object] | None = None
        if settings.stage == "stage4_qat":
            quantization_raw = config.values.get("quantization")
            quantization_values = (
                cast(dict[str, object], quantization_raw)
                if isinstance(quantization_raw, dict)
                else {}
            )
            calibration_batches = int(str(quantization_values.get("calibration_batches", 100)))
            if calibration_batches <= 0:
                raise ValueError("quantization.calibration_batches must be positive")
            set_qat_fake_quant(runtime.raw_model, False)
            calibrator = ActivationCalibrator.attach(runtime.raw_model)
            try:
                runtime.raw_model.eval()
                representative = runtime.train_batches(start_epoch)[:calibration_batches]
                with torch.no_grad():
                    for raw_batch in representative:
                        runtime.loss(runtime.raw_model, raw_batch, trainer.global_step)
            finally:
                calibrator.close()
                set_qat_fake_quant(runtime.raw_model, True)
            ranges_by_rank = context.all_gather_mappings(calibrator.report())
            quantization_report = {
                **quantization_inventory(runtime.raw_model),
                "calibration_split": "train",
                "calibration_batches_per_rank": len(representative),
                "calibration_by_rank": list(ranges_by_rank),
                "weight_quantization": "symmetric_int8_per_output_channel",
                "activation_quantization": "reported_not_rewritten",
            }
            if writer is not None:
                writer.write(
                    {
                        "event": "quantization_calibration",
                        "stage": settings.stage,
                        "calibration_batches": len(representative),
                        "observed_subgraphs": len(calibrator.ranges),
                    }
                )

        latest = run_dir / "checkpoints" / "latest.pt"
        best = run_dir / "checkpoints" / "best.pt"

        def event_callback(event: Mapping[str, object]) -> None:
            if writer is not None:
                writer.write({"stage": settings.stage, **dict(event)})

        current_epoch = start_epoch

        def checkpoint(epoch: int, next_batch: int) -> None:
            rng_states = context.all_gather_mappings(capture_rng_state())
            save_checkpoint(
                latest,
                runtime.model,
                optimizer,
                scheduler=scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
                step=trainer.global_step,
                next_batch_index=next_batch,
                optimizer_step=trainer.optimizer_step,
                best_metric=best_metric,
                sampler_state={"epoch": epoch, "seed": settings.seed},
                config=config.values,
                data_manifest_hash=corpus_digest,
                model_lock_hash=runtime.model_lock_digest,
                cache_manifest_hash=runtime.cache_manifest_sha256,
                run_manifest_hash=run_manifest_digest,
                git_revision=git_revision,
                rng_states=rng_states,
                rank=context.rank,
            )
            context.barrier()

        def step_callback(next_batch: int, optimizer_step: int) -> None:
            if optimizer_step % settings.checkpoint_interval_optimizer_steps == 0:
                checkpoint(current_epoch, next_batch)

        invocation_start_step = trainer.optimizer_step
        status = "complete"
        for epoch in range(start_epoch, settings.epochs):
            current_epoch = epoch
            runtime.set_epoch(epoch)
            train_batches = runtime.train_batches(epoch)
            local_limit = None
            if stop_after_optimizer_steps is not None:
                remaining = stop_after_optimizer_steps - (
                    trainer.optimizer_step - invocation_start_step
                )
                if remaining <= 0:
                    status = "stopped"
                    break
                local_limit = remaining
            train_result = trainer.train_epoch(
                train_batches,
                runtime.loss,
                start_batch=start_batch if epoch == start_epoch else 0,
                max_optimizer_steps=local_limit,
                event_callback=event_callback,
                step_callback=step_callback,
            )
            if writer is not None:
                writer.write(
                    {
                        "event": "train_epoch",
                        "stage": settings.stage,
                        "epoch": epoch,
                        "mean_loss": train_result.mean_loss,
                        "examples": train_result.examples,
                        "micro_batches": train_result.micro_batches,
                        "optimizer_steps": train_result.optimizer_steps,
                        **dict(train_result.metrics),
                    }
                )
            if not train_result.complete:
                checkpoint(epoch, train_result.next_batch_index)
                status = "stopped"
                start_batch = train_result.next_batch_index
                break
            start_batch = 0
            validation_loss: float | None = None
            if (epoch + 1) % settings.validation_interval_epochs == 0:
                validation_batches = runtime.validation_batches(epoch)
                if validation_batches or settings.validation_required:
                    validation = trainer.validate(validation_batches, runtime.loss)
                    validation_loss = validation.mean_loss
                    if writer is not None:
                        writer.write(
                            {
                                "event": "validation_epoch",
                                "stage": settings.stage,
                                "epoch": epoch,
                                "mean_loss": validation.mean_loss,
                                "examples": validation.examples,
                                **dict(validation.metrics),
                            }
                        )
            improved = validation_loss is not None and (
                best_metric is None or validation_loss < best_metric
            )
            if improved:
                best_metric = validation_loss
            checkpoint(epoch + 1, 0)
            if improved:
                rng_states = context.all_gather_mappings(capture_rng_state())
                save_checkpoint(
                    best,
                    runtime.model,
                    optimizer,
                    scheduler=scheduler,
                    scaler=trainer.scaler,
                    epoch=epoch + 1,
                    step=trainer.global_step,
                    optimizer_step=trainer.optimizer_step,
                    best_metric=best_metric,
                    sampler_state={"epoch": epoch + 1, "seed": settings.seed},
                    config=config.values,
                    data_manifest_hash=corpus_digest,
                    model_lock_hash=runtime.model_lock_digest,
                    cache_manifest_hash=runtime.cache_manifest_sha256,
                    run_manifest_hash=run_manifest_digest,
                    git_revision=git_revision,
                    rng_states=rng_states,
                    rank=context.rank,
                )
                context.barrier()
            if stop_after_optimizer_steps is not None and (
                trainer.optimizer_step - invocation_start_step >= stop_after_optimizer_steps
            ):
                status = "stopped" if epoch + 1 < settings.epochs else "complete"
                break
        if settings.stage == "stage4_qat" and status == "complete":
            if quantization_report is None:
                raise RuntimeError("Stage-4 quantization report was not initialized")
            comparison_batches = runtime.validation_batches(current_epoch)
            qat_metrics = _evaluate_quantization_variant(
                runtime.model,
                comparison_batches,
                runtime.loss,
                context=context,
                step=trainer.global_step,
            )
            fp_model = deepcopy(runtime.raw_model).to(context.device)
            set_qat_fake_quant(fp_model, False)
            fp_metrics = _evaluate_quantization_variant(
                fp_model,
                comparison_batches,
                runtime.loss,
                context=context,
                step=trainer.global_step,
            )
            del fp_model
            ptq_model = deepcopy(runtime.raw_model).to(context.device)
            converted_layers = convert_per_channel_ptq(ptq_model)
            ptq_metrics = _evaluate_quantization_variant(
                ptq_model,
                comparison_batches,
                runtime.loss,
                context=context,
                step=trainer.global_step,
            )
            del ptq_model
            quantization_report["comparison"] = {
                "same_checkpoint_and_validation_data": True,
                "fp32": fp_metrics,
                "ptq_weight_only": ptq_metrics,
                "qat_fake_quant": qat_metrics,
                "ptq_converted_layers": converted_layers,
            }
            if context.is_main:
                (run_dir / "quantization-report.json").write_text(
                    json.dumps(quantization_report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            context.barrier()
        if settings.stage == "stage1_semantics_graph" and status == "complete":
            visual = getattr(runtime.raw_model, "visual", None)
            if not isinstance(visual, nn.Module):
                raise RuntimeError("Stage-1 semantics model is missing its visual branch")
            visual_checkpoint = run_dir / "checkpoints" / "visual-model.pt"
            if context.is_main:
                temporary = visual_checkpoint.with_suffix(".pt.partial")
                torch.save(
                    {
                        "format_version": 1,
                        "model": visual.state_dict(),
                        "source_stage": "stage1_semantics_graph",
                        "data_manifest_digest": corpus_digest,
                        "model_lock_digest": runtime.model_lock_digest,
                    },
                    temporary,
                )
                os.replace(temporary, visual_checkpoint)
                if writer is not None:
                    writer.write(
                        {
                            "event": "visual_checkpoint",
                            "stage": settings.stage,
                            "path": visual_checkpoint.as_posix(),
                        }
                    )
            context.barrier()
        epochs_completed = current_epoch + (1 if status == "complete" or start_batch == 0 else 0)
        if writer is not None:
            writer.write(
                {
                    "event": "run_finished",
                    "stage": settings.stage,
                    "status": status,
                    "epochs_completed": epochs_completed,
                    "global_step": trainer.global_step,
                    "optimizer_step": trainer.optimizer_step,
                    "best_validation_loss": best_metric,
                }
            )
        context.barrier()
        return StageRunResult(
            status,
            settings.stage,
            run_dir.as_posix(),
            context.rank,
            context.world_size,
            str(context.device),
            precision,
            epochs_completed,
            trainer.global_step,
            trainer.optimizer_step,
            best_metric,
            latest.as_posix() if latest.is_file() else None,
            best.as_posix() if best.is_file() else None,
            plan.effective_global_batch,
            plan.accumulation_steps,
            int(runtime.trainability["trainable_parameters"]),
            int(runtime.trainability["frozen_parameters"]),
            corpus_digest,
            runtime.model_lock_digest,
            runtime.cache_manifest_sha256,
            run_manifest_digest,
        )
    finally:
        if writer is not None:
            writer.close()
        context.cleanup()


def stage_run_result_dict(result: StageRunResult | NativeStageResult) -> dict[str, Any]:
    """Expose one JSON-safe conversion for module launchers."""

    return asdict(result)


def probe_batch_size(
    config: ResolvedConfig,
    *,
    stage: str,
    manifest: Path,
    cache_manifest: Path | None = None,
    model_lock: Path | None = None,
    model_cache_root: Path | None = None,
    device: str = "cpu",
    maximum_batch_size: int = 64,
    max_commands: int | None = None,
) -> BatchProbeResult:
    """Probe real forward/backward memory and surface, rather than suppress, CUDA OOM."""

    if maximum_batch_size <= 0:
        raise ValueError("maximum_batch_size must be positive")
    base = StageRunSettings.from_values(
        cast(Mapping[str, object], config.values),
        stage_name=stage,
        max_commands=max_commands,
    )
    context = initialize_distributed(device)
    if context.distributed:
        context.cleanup()
        raise ValueError("train probe-batch must run as one process on the selected device")
    attempted: list[int] = []
    peaks: dict[int, int] = {}
    first_oom: int | None = None
    oom_message: str | None = None
    root = repository_root()

    def attempt(batch_size: int) -> bool:
        nonlocal first_oom, oom_message
        attempted.append(batch_size)
        settings = replace(
            base,
            per_device_batch_size=batch_size,
            global_batch_size=batch_size * context.world_size,
            tensorboard=False,
        )
        seed_everything(settings.seed + context.rank)
        runtime: _StageRuntime | None = None
        try:
            if context.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(context.device)
            runtime = _build_runtime(
                config,
                settings,
                context=context,
                manifest=manifest,
                cache_manifest=cache_manifest,
                model_lock=model_lock,
                model_cache_root=(model_cache_root or root / ".cache" / "models"),
            )
            runtime.set_epoch(0)
            optimizer = build_adamw(
                runtime.raw_model,
                pretrained_prefixes=runtime.pretrained_prefixes,
                new_head_lr=settings.new_head_lr,
                pretrained_lr=settings.pretrained_lr,
                weight_decay=settings.weight_decay,
            )
            precision = select_precision(context.device, settings.precision)
            trainer = Trainer(
                runtime.model,
                optimizer,
                config=TrainerConfig(
                    device=str(context.device),
                    precision=precision,
                    gradient_accumulation_steps=1,
                    grad_clip_norm=settings.gradient_clip_norm,
                    log_interval=1,
                ),
                distributed=context,
            )
            batches = runtime.train_batches(0)
            trainer.train_epoch([batches[0]], runtime.loss, max_optimizer_steps=1)
            peaks[batch_size] = (
                int(torch.cuda.max_memory_allocated(context.device))
                if context.device.type == "cuda"
                else 0
            )
            context.barrier()
            return True
        except RuntimeError as error:
            message = str(error)
            if context.device.type != "cuda" or "out of memory" not in message.casefold():
                raise
            first_oom = batch_size if first_oom is None else min(first_oom, batch_size)
            oom_message = message
            return False
        finally:
            del runtime
            gc.collect()
            if context.device.type == "cuda":
                torch.cuda.empty_cache()

    try:
        low = 0
        candidate = 1
        failed_at: int | None = None
        while candidate <= maximum_batch_size:
            if attempt(candidate):
                low = candidate
                if candidate == maximum_batch_size:
                    break
                candidate = min(candidate * 2, maximum_batch_size)
            else:
                failed_at = candidate
                break
        if low == 0:
            raise RuntimeError(
                f"batch probe OOM at per-device batch 1: {oom_message or 'unknown CUDA OOM'}"
            )
        high = (failed_at - 1) if failed_at is not None else low
        while high > low:
            midpoint = (low + high + 1) // 2
            if attempt(midpoint):
                low = midpoint
            else:
                high = midpoint - 1
        accumulation = math.ceil(256 / (low * context.world_size))
        return BatchProbeResult(
            base.stage,
            str(context.device),
            select_precision(context.device, base.precision),
            context.world_size,
            low,
            peaks[low],
            accumulation,
            accumulation * low * context.world_size,
            tuple(attempted),
            first_oom,
            oom_message,
        )
    finally:
        context.cleanup()
