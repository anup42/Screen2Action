"""Typed action-specific grounding losses and reconstructed confidence labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from screen2action.models.sparse_grounder import GroundingTensorOutput


@dataclass(frozen=True, slots=True)
class GroundingSupervision:
    candidate_indices: torch.Tensor
    candidate_mask: torch.Tensor
    point_local: torch.Tensor
    point_mask: torch.Tensor
    action_types: torch.Tensor
    action_mask: torch.Tensor
    long_press_point_local: torch.Tensor
    long_press_mask: torch.Tensor
    scroll_container_indices: torch.Tensor
    scroll_mask: torch.Tensor
    scroll_delta: torch.Tensor
    drag_source_indices: torch.Tensor
    drag_destination_indices: torch.Tensor
    drag_mask: torch.Tensor
    drag_source_point_local: torch.Tensor
    drag_destination_point_local: torch.Tensor
    drag_duration: torch.Tensor
    drag_duration_mask: torch.Tensor
    target_boxes: torch.Tensor
    target_box_mask: torch.Tensor

    def __post_init__(self) -> None:
        count = self.candidate_indices.shape[0]
        vectors = (
            self.candidate_indices,
            self.candidate_mask,
            self.point_mask,
            self.action_types,
            self.action_mask,
            self.long_press_mask,
            self.scroll_container_indices,
            self.scroll_mask,
            self.drag_source_indices,
            self.drag_destination_indices,
            self.drag_mask,
            self.drag_duration_mask,
            self.target_box_mask,
        )
        if any(value.shape != (count,) for value in vectors):
            raise ValueError("grounding vector labels/masks must have shape [C]")
        for value, width, name in (
            (self.point_local, 2, "point_local"),
            (self.long_press_point_local, 2, "long_press_point_local"),
            (self.scroll_delta, 2, "scroll_delta"),
            (self.drag_source_point_local, 2, "drag_source_point_local"),
            (self.drag_destination_point_local, 2, "drag_destination_point_local"),
            (self.target_boxes, 4, "target_boxes"),
        ):
            if value.shape != (count, width):
                raise ValueError(f"{name} has an invalid shape")
        if self.drag_duration.shape != (count, 1):
            raise ValueError("drag_duration must have shape [C,1]")
        masks = (
            self.candidate_mask,
            self.point_mask,
            self.action_mask,
            self.long_press_mask,
            self.scroll_mask,
            self.drag_mask,
            self.drag_duration_mask,
            self.target_box_mask,
        )
        if any(mask.dtype is not torch.bool for mask in masks):
            raise ValueError("grounding supervision masks must be boolean")


@dataclass(frozen=True, slots=True)
class ConfidenceReconstructionConfig:
    """Non-paper confidence-target policy `grounding_correctness_v1`."""

    enabled: bool = True
    warmup_steps: int = 0
    refresh_interval: int = 1
    loss_weight: float = 0.1
    policy_version: str = "grounding_correctness_v1"

    def __post_init__(self) -> None:
        if self.warmup_steps < 0 or self.refresh_interval <= 0 or self.loss_weight < 0.0:
            raise ValueError("invalid confidence reconstruction schedule")

    def active(self, step: int) -> bool:
        if step < 0:
            raise ValueError("training step must be non-negative")
        return self.enabled and step >= self.warmup_steps

    def should_refresh(self, step: int) -> bool:
        return self.active(step) and (step - self.warmup_steps) % self.refresh_interval == 0


@dataclass(frozen=True, slots=True)
class GroundingLoss:
    total: torch.Tensor
    candidate: torch.Tensor
    point: torch.Tensor
    action: torch.Tensor
    parameters: torch.Tensor
    confidence: torch.Tensor
    confidence_count: int


def grounding_correctness_targets(
    output: GroundingTensorOutput,
    *,
    candidate_boxes: torch.Tensor,
    supervision: GroundingSupervision,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build detached candidate-and-point correctness labels for confidence BCE."""

    command_count, candidate_count = output.candidate_logits.shape
    if candidate_boxes.shape != (command_count, candidate_count, 4):
        raise ValueError("candidate_boxes must have shape [C,K,4]")
    predicted = output.candidate_logits.detach().argmax(dim=-1)
    row = torch.arange(command_count, device=predicted.device)
    local = output.point_local.detach()[row, predicted]
    boxes = candidate_boxes.detach()[row, predicted]
    screen_points = torch.stack(
        (
            boxes[:, 0] + local[:, 0] * (boxes[:, 2] - boxes[:, 0]),
            boxes[:, 1] + local[:, 1] * (boxes[:, 3] - boxes[:, 1]),
        ),
        dim=-1,
    )
    target = supervision.target_boxes
    point_correct = (
        (screen_points[:, 0] >= target[:, 0])
        & (screen_points[:, 0] <= target[:, 2])
        & (screen_points[:, 1] >= target[:, 1])
        & (screen_points[:, 1] <= target[:, 3])
    )
    candidate_correct = predicted == supervision.candidate_indices
    mask = supervision.candidate_mask & supervision.target_box_mask
    return (candidate_correct & point_correct).to(output.candidate_logits.dtype).detach(), mask


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    return F.smooth_l1_loss(prediction[mask], target[mask]) if bool(mask.any()) else zero


def grounding_loss(
    output: GroundingTensorOutput,
    supervision: GroundingSupervision,
    *,
    candidate_boxes: torch.Tensor,
    confidence_config: ConfidenceReconstructionConfig | None = None,
    step: int = 0,
    candidate_weight: float = 1.0,
    point_weight: float = 1.0,
    action_weight: float = 0.4,
) -> GroundingLoss:
    """Apply losses only to parameters valid for each supervised action type."""

    zero = output.point_local.sum() * 0.0
    candidate = (
        F.cross_entropy(
            output.candidate_logits[supervision.candidate_mask],
            supervision.candidate_indices[supervision.candidate_mask],
        )
        if bool(supervision.candidate_mask.any())
        else zero
    )
    rows = torch.arange(output.candidate_logits.shape[0], device=output.candidate_logits.device)
    safe_candidate = supervision.candidate_indices.clamp(0, output.candidate_logits.shape[1] - 1)
    action_mask = supervision.action_mask & supervision.candidate_mask
    action = (
        F.cross_entropy(
            output.action_type_logits[rows, safe_candidate][action_mask],
            supervision.action_types[action_mask],
        )
        if bool(action_mask.any())
        else zero
    )
    click_mask = supervision.point_mask & action_mask & (supervision.action_types == 0)
    click_points = output.point_local[rows, safe_candidate]
    point = _masked_smooth_l1(click_points, supervision.point_local, click_mask, zero)

    long_mask = supervision.long_press_mask & action_mask & (supervision.action_types == 3)
    long_points = output.long_press_point_local[rows, safe_candidate]
    long_loss = _masked_smooth_l1(long_points, supervision.long_press_point_local, long_mask, zero)

    scroll_mask = (
        supervision.scroll_mask & supervision.action_mask & (supervision.action_types == 2)
    )
    scroll_container = (
        F.cross_entropy(
            output.scroll_container_logits[scroll_mask],
            supervision.scroll_container_indices[scroll_mask],
        )
        if bool(scroll_mask.any())
        else zero
    )
    scroll_delta = output.scroll_delta[
        rows, supervision.scroll_container_indices.clamp(0, output.candidate_logits.shape[1] - 1)
    ]
    scroll_delta_loss = _masked_smooth_l1(scroll_delta, supervision.scroll_delta, scroll_mask, zero)

    drag_mask = supervision.drag_mask & supervision.action_mask & (supervision.action_types == 1)
    drag_source = (
        F.cross_entropy(
            output.drag_source_logits[drag_mask],
            supervision.drag_source_indices[drag_mask],
        )
        if bool(drag_mask.any())
        else zero
    )
    drag_destination = (
        F.cross_entropy(
            output.drag_destination_logits[drag_mask],
            supervision.drag_destination_indices[drag_mask],
        )
        if bool(drag_mask.any())
        else zero
    )
    safe_source = supervision.drag_source_indices.clamp(0, output.candidate_logits.shape[1] - 1)
    safe_destination = supervision.drag_destination_indices.clamp(
        0, output.candidate_logits.shape[1] - 1
    )
    drag_source_point = output.drag_source_point_local[rows, safe_source]
    drag_destination_point = output.drag_destination_point_local[rows, safe_destination]
    drag_source_point_loss = _masked_smooth_l1(
        drag_source_point, supervision.drag_source_point_local, drag_mask, zero
    )
    drag_destination_point_loss = _masked_smooth_l1(
        drag_destination_point,
        supervision.drag_destination_point_local,
        drag_mask,
        zero,
    )
    duration_mask = supervision.drag_duration_mask & drag_mask
    drag_duration = output.drag_duration[rows, safe_source]
    duration_loss = _masked_smooth_l1(drag_duration, supervision.drag_duration, duration_mask, zero)
    parameters = (
        long_loss
        + scroll_container
        + scroll_delta_loss
        + drag_source
        + drag_destination
        + drag_source_point_loss
        + drag_destination_point_loss
        + duration_loss
    )

    config = confidence_config or ConfidenceReconstructionConfig(enabled=False)
    confidence = zero
    confidence_count = 0
    if config.should_refresh(step):
        targets, confidence_mask = grounding_correctness_targets(
            output,
            candidate_boxes=candidate_boxes,
            supervision=supervision,
        )
        predicted = output.candidate_logits.detach().argmax(dim=-1)
        confidence_logits = output.confidence_logits[rows, predicted]
        if bool(confidence_mask.any()):
            confidence = F.binary_cross_entropy_with_logits(
                confidence_logits[confidence_mask], targets[confidence_mask]
            )
            confidence_count = int(confidence_mask.sum())
    total = (
        candidate_weight * candidate
        + point_weight * point
        + action_weight * (action + parameters)
        + config.loss_weight * confidence
    )
    return GroundingLoss(
        total,
        candidate,
        point,
        action,
        parameters,
        confidence,
        confidence_count,
    )
