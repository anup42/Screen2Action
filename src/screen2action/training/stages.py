"""Stage definitions and freeze contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One reference training stage."""

    name: str
    epochs: int
    freeze_ocr: bool
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
    "stage1_perception": StageSpec("stage1_perception", 12, False),
    "stage2_selector_retrieval": StageSpec(
        "stage2_selector_retrieval",
        8,
        False,
        forced_positive_probability=0.5,
    ),
    "stage3_joint": StageSpec("stage3_joint", 6, True),
    "stage4_qat": StageSpec("stage4_qat", 3, True, new_head_lr=2e-5, pretrained_lr=2e-6),
}


def stage_spec(name: str) -> StageSpec:
    """Return a named stage or raise an actionable error."""

    try:
        return STAGES[name]
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


def freeze_ocr_parameters(model: object) -> None:
    """Freeze a module named `ocr` when present without requiring a perception package."""

    ocr = getattr(model, "ocr", None)
    if ocr is None or not hasattr(ocr, "parameters"):
        return
    for parameter in ocr.parameters():
        parameter.requires_grad_(False)
