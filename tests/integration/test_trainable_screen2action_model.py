"""Unified trainable multi-screen/multi-command model integration tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from screen2action.data.schema import (
    EdgeRecord,
    NodeRecord,
    NodeType,
    RelationDirection,
    RelationType,
)
from screen2action.models.batching import CommandBatch
from screen2action.models.screen2action_model import (
    Screen2ActionBatch,
    Screen2ActionModel,
    Screen2ActionModelConfig,
)
from screen2action.perception.base import normalize_image
from screen2action.perception.pipeline import PerceptionFrame
from screen2action.runtime.model_runtime import Screen2ActionInferenceRuntime

pytestmark = pytest.mark.integration_model


def _frame(image: torch.Tensor, suffix: int) -> PerceptionFrame:
    normalized = normalize_image(image)
    nodes = (
        NodeRecord(
            0,
            NodeType.ROOT,
            (0.0, 0.0, 1.0, 1.0),
            detector_confidence=1.0,
            mandatory=True,
            actionability_mask=(False, False, False, False),
            annotation_source="fixture",
        ),
        NodeRecord(
            1,
            NodeType.CONTROL,
            (0.05, 0.1, 0.45, 0.4),
            detector_confidence=0.9,
            text_token_ids=(2 + suffix, 4),
            ocr_confidence=0.8,
            roi_visual_feature=(0.1,) * 16,
            visual_feature_confidence=0.9,
            actionability_logits=(5.0, -2.0, -2.0, -2.0),
            parent_id=0,
            annotation_source="fixture",
            provenance={"original_class_id": "2"},
        ),
        NodeRecord(
            2,
            NodeType.CONTROL,
            (0.55, 0.5, 0.95, 0.9),
            detector_confidence=0.85,
            text_token_ids=(3 + suffix, 5),
            ocr_confidence=0.75,
            roi_visual_feature=(0.2,) * 16,
            visual_feature_confidence=0.8,
            actionability_logits=(4.0, -2.0, -2.0, -2.0),
            parent_id=0,
            annotation_source="fixture",
            provenance={"original_class_id": "2"},
        ),
    )
    edges = (
        EdgeRecord(0, 1, RelationType.CONTAINMENT, RelationDirection.CONTAINS, (0.1,) * 12),
        EdgeRecord(0, 2, RelationType.CONTAINMENT, RelationDirection.CONTAINS, (0.2,) * 12),
        EdgeRecord(1, 2, RelationType.PROXIMITY, RelationDirection.RIGHT, (0.3,) * 12),
    )
    return PerceptionFrame(
        normalized.sha256,
        normalized.width,
        normalized.height,
        nodes,
        edges,
        0,
        "fixture",
        {"suffix": suffix},
    )


def _commands() -> CommandBatch:
    input_ids = torch.tensor([[1, 2, 4, 0], [1, 3, 5, 0], [1, 4, 5, 0]])
    count = input_ids.shape[0]
    return CommandBatch(
        input_ids=input_ids,
        attention_mask=input_ids != 0,
        screen_indices=torch.tensor([0, 0, 1]),
        target_node_ids=torch.tensor([1, 2, 1]),
        target_mask=torch.ones(count, dtype=torch.bool),
        reference_node_ids=torch.zeros((count, 1), dtype=torch.long),
        reference_mask=torch.zeros((count, 1), dtype=torch.bool),
        action_types=torch.zeros(count, dtype=torch.long),
        action_mask=torch.ones(count, dtype=torch.bool),
        target_boxes=torch.tensor(
            [[0.05, 0.1, 0.45, 0.4], [0.55, 0.5, 0.95, 0.9], [0.05, 0.1, 0.45, 0.4]]
        ),
        target_box_mask=torch.ones(count, dtype=torch.bool),
        target_points=torch.tensor([[0.25, 0.25], [0.75, 0.7], [0.25, 0.25]]),
        target_point_mask=torch.ones(count, dtype=torch.bool),
        parameter_targets=torch.zeros((count, 9)),
        parameter_mask=torch.zeros((count, 9), dtype=torch.bool),
        parameter_node_ids=torch.zeros((count, 2), dtype=torch.long),
        parameter_node_mask=torch.zeros((count, 2), dtype=torch.bool),
    )


def test_unified_model_runs_each_screen_once_then_fans_out_commands_with_gradients() -> None:
    torch.manual_seed(21)
    config = replace(Screen2ActionModelConfig.tiny_cpu(), ssb_budget=512, top_k=3)
    model = Screen2ActionModel(config=config)
    model.retention_scorer.projection.weight.data.zero_()
    model.retention_scorer.projection.bias.data.fill_(10.0)
    model.train()
    images = (
        torch.zeros((3, 32, 48), dtype=torch.uint8),
        torch.full((3, 32, 48), 64, dtype=torch.uint8),
    )
    frames = (_frame(images[0], 0), _frame(images[1], 1))

    output = model(
        Screen2ActionBatch(images, _commands(), perceived_frames=frames),
        stage="stage3_joint",
        generator=torch.Generator().manual_seed(3),
    )

    encoded = output.encoded_frames
    grounded = output.grounded_commands
    assert encoded.node_states.shape[:2] == (2, 3)
    assert encoded.valid_node_mask.shape == (2, 3)
    assert encoded.valid_edge_mask.shape == (2, 3)
    assert encoded.valid_text_token_mask.shape == (2, 3, 16)
    assert grounded.tensors.candidate_logits.shape == (3, 3)
    assert grounded.candidate_valid_mask.shape == (3, 3)
    assert grounded.valid_crop_token_mask.shape == (3, 3, 16)
    assert torch.isfinite(output.total_loss)
    output.total_loss.backward()
    assert model.node_encoder.output[0].weight.grad is not None
    assert model.command_encoder.embedding.weight.grad is not None
    assert model.retention_scorer.projection.weight.grad is not None
    assert model.grounder.candidate_head.weight.grad is not None


def test_model_level_selection_is_exactly_command_independent() -> None:
    torch.manual_seed(22)
    config = replace(Screen2ActionModelConfig.tiny_cpu(), ssb_budget=512, top_k=3)
    model = Screen2ActionModel(config=config)
    model.eval()
    images = (
        torch.zeros((3, 32, 48), dtype=torch.uint8),
        torch.full((3, 32, 48), 64, dtype=torch.uint8),
    )
    frames = (_frame(images[0], 0), _frame(images[1], 1))
    first_commands = _commands()
    second_commands = replace(
        first_commands,
        input_ids=torch.tensor([[1, 7, 7, 0], [1, 8, 8, 0], [1, 9, 9, 0]]),
    )

    first = model(
        Screen2ActionBatch(images, first_commands, perceived_frames=frames), stage="inference"
    )
    second = model(
        Screen2ActionBatch(images, second_commands, perceived_frames=frames), stage="inference"
    )

    assert torch.equal(
        first.encoded_frames.selected_node_mask,
        second.encoded_frames.selected_node_mask,
    )
    assert torch.equal(
        first.encoded_frames.selection_weights,
        second.encoded_frames.selection_weights,
    )


def test_thin_runtime_requires_caller_controlled_eval_mode() -> None:
    config = replace(Screen2ActionModelConfig.tiny_cpu(), ssb_budget=512, top_k=3)
    model = Screen2ActionModel(config=config)
    images = (
        torch.zeros((3, 32, 48), dtype=torch.uint8),
        torch.full((3, 32, 48), 64, dtype=torch.uint8),
    )
    batch = Screen2ActionBatch(
        images,
        _commands(),
        perceived_frames=(_frame(images[0], 0), _frame(images[1], 1)),
    )
    runtime = Screen2ActionInferenceRuntime(model)

    with pytest.raises(RuntimeError, match="eval mode"):
        runtime.predict(batch)
    model.eval()
    output = runtime.predict(batch)
    assert output.stage == "inference"
