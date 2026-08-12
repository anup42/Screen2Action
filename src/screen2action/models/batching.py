"""Padded command supervision and deterministic screen-grouped sampling."""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CommandBatch:
    """Multiple commands fanning out from a smaller unique-screen batch."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    screen_indices: torch.Tensor
    target_node_ids: torch.Tensor
    target_mask: torch.Tensor
    reference_node_ids: torch.Tensor
    reference_mask: torch.Tensor
    action_types: torch.Tensor
    action_mask: torch.Tensor
    target_boxes: torch.Tensor
    target_box_mask: torch.Tensor
    target_points: torch.Tensor
    target_point_mask: torch.Tensor
    parameter_targets: torch.Tensor
    parameter_mask: torch.Tensor
    parameter_node_ids: torch.Tensor
    parameter_node_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("command IDs and attention masks must have shape [C,T]")
        command_count = self.input_ids.shape[0]
        vector_fields = (
            self.screen_indices,
            self.target_node_ids,
            self.target_mask,
            self.action_types,
            self.action_mask,
            self.target_box_mask,
            self.target_point_mask,
        )
        if any(field.shape != (command_count,) for field in vector_fields):
            raise ValueError("command vector labels/masks must have shape [C]")
        if self.reference_node_ids.ndim != 2 or self.reference_node_ids.shape[0] != command_count:
            raise ValueError("reference IDs must have shape [C,R]")
        if self.reference_mask.shape != self.reference_node_ids.shape:
            raise ValueError("reference mask must match reference IDs")
        if self.target_boxes.shape != (command_count, 4):
            raise ValueError("target boxes must have shape [C,4]")
        if self.target_points.shape != (command_count, 2):
            raise ValueError("target points must have shape [C,2]")
        if self.parameter_targets.ndim != 2 or self.parameter_targets.shape[0] != command_count:
            raise ValueError("parameter targets must have shape [C,A]")
        if self.parameter_mask.shape != self.parameter_targets.shape:
            raise ValueError("parameter mask must match parameter targets")
        if self.parameter_node_ids.shape != (command_count, 2):
            raise ValueError("parameter node IDs must have shape [C,2]")
        if self.parameter_node_mask.shape != self.parameter_node_ids.shape:
            raise ValueError("parameter node mask must match parameter node IDs")
        for mask in (
            self.attention_mask,
            self.target_mask,
            self.reference_mask,
            self.action_mask,
            self.target_box_mask,
            self.target_point_mask,
            self.parameter_mask,
            self.parameter_node_mask,
        ):
            if mask.dtype is not torch.bool:
                raise ValueError("all command supervision masks must be boolean")
        if command_count and int(self.screen_indices.min()) < 0:
            raise ValueError("screen indices must be non-negative")

    @classmethod
    def unsupervised(
        cls,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        screen_indices: torch.Tensor,
        *,
        reference_slots: int = 1,
        parameter_count: int = 9,
    ) -> CommandBatch:
        """Construct an inference batch with every supervision mask disabled."""

        count = input_ids.shape[0]
        device = input_ids.device
        return cls(
            input_ids=input_ids,
            attention_mask=attention_mask.bool(),
            screen_indices=screen_indices.to(device=device, dtype=torch.long),
            target_node_ids=torch.zeros(count, dtype=torch.long, device=device),
            target_mask=torch.zeros(count, dtype=torch.bool, device=device),
            reference_node_ids=torch.zeros(
                (count, reference_slots), dtype=torch.long, device=device
            ),
            reference_mask=torch.zeros((count, reference_slots), dtype=torch.bool, device=device),
            action_types=torch.zeros(count, dtype=torch.long, device=device),
            action_mask=torch.zeros(count, dtype=torch.bool, device=device),
            target_boxes=torch.zeros((count, 4), dtype=torch.float32, device=device),
            target_box_mask=torch.zeros(count, dtype=torch.bool, device=device),
            target_points=torch.zeros((count, 2), dtype=torch.float32, device=device),
            target_point_mask=torch.zeros(count, dtype=torch.bool, device=device),
            parameter_targets=torch.zeros(
                (count, parameter_count), dtype=torch.float32, device=device
            ),
            parameter_mask=torch.zeros((count, parameter_count), dtype=torch.bool, device=device),
            parameter_node_ids=torch.zeros((count, 2), dtype=torch.long, device=device),
            parameter_node_mask=torch.zeros((count, 2), dtype=torch.bool, device=device),
        )


class ScreenGroupedSampler:
    """Yield command indices grouped by screen so frame work runs once per screen."""

    def __init__(
        self,
        screen_ids: Sequence[str],
        *,
        screens_per_batch: int = 1,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if not screen_ids or any(not screen_id for screen_id in screen_ids):
            raise ValueError("screen_ids must contain non-empty identifiers")
        if screens_per_batch <= 0:
            raise ValueError("screens_per_batch must be positive")
        grouped: OrderedDict[str, list[int]] = OrderedDict()
        for index, screen_id in enumerate(screen_ids):
            grouped.setdefault(screen_id, []).append(index)
        self.groups = tuple((screen_id, tuple(indices)) for screen_id, indices in grouped.items())
        self.screens_per_batch = screens_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.groups) + self.screens_per_batch - 1) // self.screens_per_batch

    def __iter__(self) -> Iterator[list[int]]:
        groups = list(self.groups)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        for offset in range(0, len(groups), self.screens_per_batch):
            batch = groups[offset : offset + self.screens_per_batch]
            yield [index for _, indices in batch for index in indices]
