"""Unified trainable multi-screen/multi-command model integration tests."""

from __future__ import annotations

import os
import socket
from dataclasses import replace

import pytest
import torch
from torch import nn

from screen2action.data.schema import (
    EdgeRecord,
    NodeRecord,
    NodeType,
    RelationDirection,
    RelationType,
)
from screen2action.losses.total import TotalLossWeights
from screen2action.models.batching import CommandBatch
from screen2action.models.crop_encoder import CropTokenEncoder
from screen2action.models.screen2action_model import (
    Screen2ActionBatch,
    Screen2ActionModel,
    Screen2ActionModelConfig,
)
from screen2action.perception.base import normalize_image
from screen2action.perception.pipeline import PerceptionFrame
from screen2action.runtime.model_runtime import Screen2ActionInferenceRuntime
from screen2action.training.distributed import initialize_distributed, wrap_ddp
from screen2action.training.engine import LossResult, Trainer, TrainerConfig
from screen2action.training.model_factory import build_screen2action_model

pytestmark = pytest.mark.integration_model


def _distributed_model_worker(rank: int, port: int) -> None:
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    torch.set_num_threads(1)
    context = initialize_distributed("cpu")
    try:
        torch.manual_seed(21)
        model = Screen2ActionModel(
            config=replace(Screen2ActionModelConfig.tiny_cpu(), ssb_budget=512, top_k=3)
        )
        wrapped = wrap_ddp(model, context, find_unused_parameters=True)
        images = (torch.zeros((3, 32, 48), dtype=torch.uint8),) * 2
        commands = _commands()
        if rank == 1:
            commands = replace(
                commands,
                target_mask=torch.zeros(3, dtype=torch.bool),
                action_mask=torch.zeros(3, dtype=torch.bool),
            )
        batch = Screen2ActionBatch(
            images,
            commands,
            perceived_frames=tuple(_frame(image, i) for i, image in enumerate(images)),
        )
        trainer = Trainer(
            wrapped,
            torch.optim.AdamW(model.parameters(), lr=0.001),
            config=TrainerConfig(gradient_accumulation_steps=2),
            distributed=context,
        )
        result = trainer.train_epoch(
            [batch] * 4, lambda module, value, step: LossResult(module(value).total_loss, {}, 3)
        )
        assert result.complete and trainer.optimizer_step == 2
        checksums = context.all_gather_mappings(
            {
                "sum": sum(
                    float(parameter.detach().double().sum()) for parameter in model.parameters()
                )
            }
        )
        assert checksums[0] == checksums[1]
    finally:
        context.cleanup()


def test_unified_model_supports_two_rank_accumulated_training() -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Torch Gloo distributed support is unavailable")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    workers = torch.multiprocessing.spawn(
        _distributed_model_worker, args=(port,), nprocs=2, join=False
    )
    try:
        for _ in range(6):
            if workers.join(timeout=10):
                return
        pytest.fail("two-rank model training did not finish within 60 seconds")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


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
    for module in (model.retriever, model.reranker):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert any(grad is not None and bool(grad.abs().sum() > 0) for grad in gradients)


def test_inference_candidates_do_not_depend_on_action_supervision() -> None:
    torch.manual_seed(22)
    model = Screen2ActionModel(
        config=replace(Screen2ActionModelConfig.tiny_cpu(), ssb_budget=512, top_k=1)
    ).eval()
    images = (torch.zeros((3, 32, 48), dtype=torch.uint8),) * 2
    frames = tuple(_frame(image, index) for index, image in enumerate(images))
    frames = tuple(
        replace(
            frame,
            nodes=(
                frame.nodes[0],
                replace(frame.nodes[1], actionability_logits=(-10.0, 10.0, -10.0, -10.0)),
                replace(frame.nodes[2], actionability_logits=(10.0, -10.0, -10.0, -10.0)),
            ),
        )
        for frame in frames
    )
    encoded = model.encode_frames(frames, screenshots=images)
    commands = _commands()
    click = model.ground_commands(encoded, commands)
    drag = model.ground_commands(
        encoded, replace(commands, action_types=torch.ones(3, dtype=torch.long))
    )
    assert torch.equal(click.candidate_node_ids, drag.candidate_node_ids)
    assert torch.equal(click.tensors.candidate_logits, drag.tensors.candidate_logits)


def test_action_points_are_converted_from_screen_to_their_candidate_crops() -> None:
    model = Screen2ActionModel(
        config=replace(Screen2ActionModelConfig.tiny_cpu(), ssb_budget=512, top_k=4)
    ).eval()
    images = (torch.zeros((3, 32, 48), dtype=torch.uint8),) * 2
    commands = _commands()
    parameters = commands.parameter_targets.clone()
    masks = commands.parameter_mask.clone()
    parameters[0, 7:9] = torch.tensor([0.25, 0.25])
    masks[0, 7:9] = True
    parameters[1, 2:6] = torch.tensor([0.75, 0.7, 0.25, 0.25])
    masks[1, 2:6] = True
    node_ids = torch.tensor([[0, 0], [2, 1], [0, 0]])
    node_mask = torch.tensor([[False, False], [True, True], [False, False]])
    commands = replace(
        commands,
        action_types=torch.tensor([3, 1, 0]),
        parameter_targets=parameters,
        parameter_mask=masks,
        parameter_node_ids=node_ids,
        parameter_node_mask=node_mask,
    )
    encoded = model.encode_frames(
        tuple(_frame(image, i) for i, image in enumerate(images)), screenshots=images
    )
    grounded = model.ground_commands(encoded, commands)
    supervision = model._grounding_supervision(grounded, commands)
    assert supervision.long_press_mask[0]
    assert supervision.drag_mask[1]
    for actual in (
        supervision.long_press_point_local[0],
        supervision.drag_source_point_local[1],
        supervision.drag_destination_point_local[1],
    ):
        torch.testing.assert_close(actual, torch.tensor([0.5, 0.5]))
    missing_destination = replace(
        commands, parameter_node_mask=node_mask & torch.tensor([True, False])
    )
    assert not model._grounding_supervision(grounded, missing_destination).drag_mask[1]
    missing_target = replace(commands, target_mask=torch.zeros_like(commands.target_mask))
    assert not model._grounding_supervision(grounded, missing_target).long_press_mask.any()


def test_frozen_crop_encoder_preserves_batchnorm_state_and_mode() -> None:
    crop = nn.Sequential(
        nn.BatchNorm2d(3), CropTokenEncoder(embedding_dim=64, token_count=16, tiny=True)
    )
    model = Screen2ActionModel(
        config=replace(Screen2ActionModelConfig.tiny_cpu(), ssb_budget=512), crop_encoder=crop
    )
    model.set_crop_encoder_updates_frozen(True)
    model.train()
    images = (torch.full((3, 32, 48), 128, dtype=torch.uint8),) * 2
    encoded = model.encode_frames(
        tuple(_frame(image, i) for i, image in enumerate(images)), screenshots=images
    )
    before = crop[0].running_mean.clone()
    model.ground_commands(encoded, _commands())
    assert torch.equal(crop[0].running_mean, before)
    assert crop.training
    assert crop[0].num_batches_tracked == 0


def test_objective_weights_and_temperature_are_applied_by_the_factory(tmp_path) -> None:
    bundle = build_screen2action_model(
        {
            "training": {
                "loss_weights": {"ui": 0.75, "survive": 0.0, "budget": 0.0},
                "ui_contrastive_temperature": 0.2,
            }
        },
        tokenizer_vocab_size=32,
        model_lock_path=None,
        cache_root=tmp_path,
    )
    assert bundle.model.loss_weights == replace(TotalLossWeights(), ui=0.75, survive=0, budget=0)
    assert bundle.model.contrastive_temperature == 0.2


def test_retrieval_learns_even_when_stochastic_selection_drops_the_target() -> None:
    model = Screen2ActionModel(config=replace(Screen2ActionModelConfig.tiny_cpu(), ssb_budget=512))
    with torch.no_grad():
        model.retention_scorer.projection.weight.zero_()
        model.retention_scorer.projection.bias.fill_(-30)
    images = (torch.zeros((3, 32, 48), dtype=torch.uint8),) * 2
    output = model(
        Screen2ActionBatch(
            images,
            _commands(),
            perceived_frames=tuple(_frame(image, i) for i, image in enumerate(images)),
        )
    )
    assert not output.loss_inputs.grounding.candidate_mask.any()
    assert output.grounded_commands.retrieval_loss > 0
    output.total_loss.backward()
    assert model.retriever.query_projection.weight.grad.abs().sum() > 0


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


def test_stage2_forced_positive_insertion_overrides_same_screen_top_k() -> None:
    config = replace(Screen2ActionModelConfig.tiny_cpu(), ssb_budget=512, top_k=1)
    model = Screen2ActionModel(config=config)
    model.retention_scorer.projection.weight.data.zero_()
    model.retention_scorer.projection.bias.data.fill_(10.0)
    model.train()
    image = torch.zeros((3, 32, 48), dtype=torch.uint8)
    commands = replace(
        _commands(),
        input_ids=_commands().input_ids[:1],
        attention_mask=_commands().attention_mask[:1],
        screen_indices=torch.zeros(1, dtype=torch.long),
        target_node_ids=torch.zeros(1, dtype=torch.long),
        target_mask=torch.ones(1, dtype=torch.bool),
        reference_node_ids=torch.zeros((1, 1), dtype=torch.long),
        reference_mask=torch.zeros((1, 1), dtype=torch.bool),
        action_types=torch.zeros(1, dtype=torch.long),
        action_mask=torch.zeros(1, dtype=torch.bool),
        target_boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        target_box_mask=torch.ones(1, dtype=torch.bool),
        target_points=torch.zeros((1, 2)),
        target_point_mask=torch.zeros(1, dtype=torch.bool),
        parameter_targets=torch.zeros((1, 9)),
        parameter_mask=torch.zeros((1, 9), dtype=torch.bool),
        parameter_node_ids=torch.zeros((1, 2), dtype=torch.long),
        parameter_node_mask=torch.zeros((1, 2), dtype=torch.bool),
    )

    output = model(
        Screen2ActionBatch((image,), commands, perceived_frames=(_frame(image, 0),)),
        stage="stage2_grounding",
        forced_positive_probability=1.0,
        generator=torch.Generator().manual_seed(4),
    )

    assert output.grounded_commands.forced_positive_mask.tolist() == [True]
    assert output.grounded_commands.candidate_node_ids.tolist() == [[0]]
