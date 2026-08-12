"""Reference stage definitions, schedules, and explicit freeze contracts."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One paper or public-reconstruction training stage."""

    name: str
    epochs: int
    mode: str
    freeze_ocr: bool
    requires_perception_cache: bool
    global_batch_size: int = 256
    new_head_lr: float = 2e-4
    pretrained_lr: float = 2e-5
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    gradient_clip_norm: float = 1.0
    gumbel_temperature_start: float = 1.0
    gumbel_temperature_end: float = 0.1
    gumbel_anneal_fraction: float = 0.8
    forced_positive_probability: float = 0.0


STAGES = {
    "stage1_semantics_graph": StageSpec(
        "stage1_semantics_graph", 12, "semantics_graph", False, False
    ),
    "stage1_detector": StageSpec("stage1_detector", 12, "detector_native", False, False),
    "stage1_ocr": StageSpec("stage1_ocr", 12, "ocr_native", False, False),
    "stage1_full": StageSpec("stage1_full", 12, "orchestration", False, False),
    "stage2_selector_retrieval": StageSpec(
        "stage2_selector_retrieval",
        8,
        "grounding",
        True,
        True,
        forced_positive_probability=0.5,
    ),
    "stage3_joint": StageSpec("stage3_joint", 6, "joint", True, True),
    "stage4_qat": StageSpec(
        "stage4_qat",
        3,
        "quantization",
        True,
        True,
        new_head_lr=2e-5,
        pretrained_lr=2e-6,
    ),
}

_ALIASES = {"stage1_perception": "stage1_semantics_graph"}


def stage_spec(name: str) -> StageSpec:
    """Return a named stage or raise an actionable error."""

    canonical = _ALIASES.get(name, name)
    try:
        return STAGES[canonical]
    except KeyError as error:
        raise ValueError(f"unknown training stage {name!r}; choose {sorted(STAGES)}") from error


def forced_positive_probability(stage_name: str, epoch: int) -> float:
    """Return Stage 2's first-two-epoch positive-candidate insertion rate."""

    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    stage = stage_spec(stage_name)
    if stage.name == "stage2_selector_retrieval" and epoch < 2:
        return stage.forced_positive_probability
    return 0.0


def gumbel_temperature(stage_name: str, *, progress: float) -> float:
    """Anneal Stage 2/3 temperature linearly over the configured first 80%."""

    if not 0.0 <= progress <= 1.0:
        raise ValueError("stage progress must be in [0, 1]")
    stage = stage_spec(stage_name)
    fraction = min(progress / max(stage.gumbel_anneal_fraction, 1e-9), 1.0)
    return stage.gumbel_temperature_start + fraction * (
        stage.gumbel_temperature_end - stage.gumbel_temperature_start
    )


def freeze_ocr_parameters(model: object) -> None:
    """Freeze every explicitly named OCR module without importing perception."""

    for name in ("ocr", "recognizer", "ocr_model"):
        module = getattr(model, name, None)
        if module is None or not hasattr(module, "parameters"):
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def configure_stage_trainability(model: nn.Module, stage_name: str) -> dict[str, int]:
    """Apply the public stage freeze policy and report trainable/frozen counts."""

    stage = stage_spec(stage_name)
    if stage.mode in {"detector_native", "ocr_native", "orchestration"}:
        raise ValueError(f"{stage.name} uses its explicit native/orchestration runner")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if stage.freeze_ocr:
        freeze_ocr_parameters(model)
        for name, parameter in model.named_parameters():
            if any(part in name.casefold() for part in ("ocr", "recognizer")):
                parameter.requires_grad_(False)
    if stage.name == "stage2_selector_retrieval":
        perception = getattr(model, "perception_service", None)
        if perception is not None:
            for component_name in ("detector", "recognizer", "visual_model"):
                component = getattr(perception, component_name, None)
                if isinstance(component, nn.Module):
                    for parameter in component.parameters():
                        parameter.requires_grad_(False)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    if trainable == 0:
        raise ValueError(f"stage {stage.name} has no trainable parameters")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}
