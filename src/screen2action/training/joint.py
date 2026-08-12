"""Stage-3 weighted differentiable downstream and semantic-perception branches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from screen2action.models.screen2action_model import (
    Screen2ActionBatch,
    Screen2ActionModel,
    Screen2ActionModelOutput,
)
from screen2action.training.data import CachedScreenBatchBuilder, TrainingCommand
from screen2action.training.semantics import (
    SemanticBatch,
    SemanticBatchBuilder,
    SemanticScreenExample,
    SemanticsGraphModel,
    SemanticsGraphOutput,
)


@dataclass(frozen=True, slots=True)
class JointStage3Batch:
    downstream: Screen2ActionBatch
    semantics: SemanticBatch | None
    examples: int


class JointStage3BatchBuilder:
    """Pair cached command data with source-supervised screens when available."""

    def __init__(
        self,
        downstream: CachedScreenBatchBuilder,
        semantics: SemanticBatchBuilder,
        semantic_by_screen: Mapping[str, SemanticScreenExample],
    ) -> None:
        self.downstream = downstream
        self.semantics = semantics
        self.semantic_by_screen = semantic_by_screen

    def build(self, commands: Sequence[TrainingCommand]) -> JointStage3Batch:
        screen_ids = tuple(dict.fromkeys(command.screen_id for command in commands))
        semantic_examples = tuple(
            self.semantic_by_screen[screen_id]
            for screen_id in screen_ids
            if screen_id in self.semantic_by_screen
        )
        return JointStage3Batch(
            self.downstream.build(commands),
            self.semantics.build(semantic_examples) if semantic_examples else None,
            len(commands),
        )


@dataclass(frozen=True, slots=True)
class JointStage3Output:
    downstream: Screen2ActionModelOutput
    semantics: SemanticsGraphOutput | None
    semantic_loss: torch.Tensor
    total_loss: torch.Tensor


class JointStage3Model(nn.Module):
    """Keep discrete proposals fixed while training non-OCR branches with named losses."""

    def __init__(
        self,
        downstream: Screen2ActionModel,
        semantics: SemanticsGraphModel,
        *,
        semantic_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if semantic_weight < 0.0:
            raise ValueError("stage3 semantic perception weight cannot be negative")
        self.downstream = downstream
        self.semantics = semantics
        self.semantic_weight = semantic_weight

    def forward(
        self,
        batch: JointStage3Batch,
        *,
        step: int,
        selector_temperature: float,
    ) -> JointStage3Output:
        downstream = self.downstream(
            batch.downstream,
            stage="stage3_joint",
            step=step,
            selector_temperature=selector_temperature,
            forced_positive_probability=0.0,
        )
        if batch.semantics is not None:
            semantics = self.semantics(batch.semantics)
            semantic_loss = semantics.total_loss
        else:
            semantics = None
            semantic_loss = sum(
                (parameter.sum() * 0.0 for parameter in self.semantics.parameters()),
                downstream.total_loss.new_zeros(()),
            )
        total = downstream.total_loss + self.semantic_weight * semantic_loss
        return JointStage3Output(downstream, semantics, semantic_loss, total)
