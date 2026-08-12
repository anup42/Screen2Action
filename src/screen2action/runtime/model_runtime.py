"""Thin inference-only wrapper around the unified trainable model."""

from __future__ import annotations

from typing import cast

import torch

from screen2action.models.screen2action_model import (
    Screen2ActionBatch,
    Screen2ActionModel,
    Screen2ActionModelOutput,
)


class Screen2ActionInferenceRuntime:
    """Run the model's explicit inference stage without mutating its mode."""

    def __init__(self, model: Screen2ActionModel) -> None:
        self.model = model

    def predict(self, batch: Screen2ActionBatch) -> Screen2ActionModelOutput:
        if self.model.training:
            raise RuntimeError("set the model to eval mode before inference runtime use")
        with torch.inference_mode():
            return cast(Screen2ActionModelOutput, self.model(batch, stage="inference"))
