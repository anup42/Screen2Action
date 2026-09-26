"""Paper-profile graph, adapters, retention, and sparse-head contracts."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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
from screen2action.models.bert_adapter import CompactBertAdapter
from screen2action.models.grounding_losses import (
    ConfidenceReconstructionConfig,
    GroundingSupervision,
    grounding_loss,
)
from screen2action.models.mobilevit import MobileVitSCropEncoder
from screen2action.models.node_encoder import NodeEncoder, node_batch_from_records
from screen2action.models.relation_gat import RelationAwareGraphAttention
from screen2action.models.retention import RetentionScorer, retention_loss
from screen2action.models.retriever import RelationAwareReranker, select_top_k_actionable
from screen2action.models.sparse_grounder import (
    SparseCandidateGrounder,
    sinusoidal_box_encoding,
)


def test_node_fusion_consumes_every_declared_modality_and_shared_text_embedding() -> None:
    shared = nn.Embedding(128, 16)
    encoder = NodeEncoder(
        embedding_dim=16,
        vocab_size=128,
        visual_feature_dim=8,
        text_embedding=shared,
    )
    nodes = (
        NodeRecord(
            0,
            NodeType.CONTROL,
            (0.1, 0.2, 0.5, 0.6),
            detector_confidence=0.9,
            text_token_ids=(3, 4),
            ocr_confidence=0.8,
            icon_probabilities=(1.0,) + (0.0,) * 86,
            icon_confidence=0.7,
            roi_visual_feature=(0.25,) * 8,
            visual_feature_confidence=0.6,
            actionability_logits=(2.0, -2.0, 0.0, 1.0),
            hierarchy_depth=2,
            provenance={"original_class_id": "2"},
        ),
        NodeRecord(1, NodeType.OTHER, (0.6, 0.1, 0.9, 0.3), actionability_mask=(False,) * 4),
    )
    batch = node_batch_from_records(nodes, visual_feature_dim=8)

    output = encoder(batch)

    assert encoder.text_embedding is shared
    assert output.shape == (2, 16)
    assert not torch.allclose(output[0], output[1])
    output.sum().backward()
    assert encoder.original_class_embedding.weight.grad is not None
    assert encoder.confidence_projection[0].weight.grad is not None
    assert shared.weight.grad is not None


@pytest.mark.parametrize("variant", ["paper_eq_v1", "cpu_reconstruction_v1"])
def test_batched_edge_and_dense_graph_paths_have_numerical_parity(variant: str) -> None:
    torch.manual_seed(12)
    layer = RelationAwareGraphAttention(
        embedding_dim=16,
        heads=4,
        variant=variant,
    )
    nodes = torch.randn(2, 4, 16, requires_grad=True)
    edge_index = torch.tensor(
        [
            [[0, 1, 0], [1, 2, 0]],
            [[0, 2, 0], [2, 1, 0]],
        ]
    )
    relation = torch.tensor([[0, 1, 0], [2, 1, 0]])
    geometry = torch.randn(2, 3, 12)
    valid_nodes = torch.tensor([[True, True, True, False], [True, True, True, False]])
    valid_edges = torch.tensor([[True, True, False], [True, True, False]])

    edge_output = layer.forward_batched_edges(
        nodes,
        edge_index,
        relation,
        geometry,
        valid_nodes=valid_nodes,
        valid_edges=valid_edges,
    )
    dense_relation = torch.full((2, 4, 4), -1, dtype=torch.long)
    dense_geometry = torch.zeros((2, 4, 4, 12))
    for batch in range(2):
        for edge in range(2):
            src = int(edge_index[batch, 0, edge])
            dst = int(edge_index[batch, 1, edge])
            dense_relation[batch, src, dst] = relation[batch, edge]
            dense_geometry[batch, src, dst] = geometry[batch, edge]
    dense_output = layer.forward_dense(nodes, dense_relation, dense_geometry, valid_nodes)

    assert torch.allclose(edge_output, dense_output, atol=1e-6, rtol=1e-6)
    assert torch.equal(edge_output[:, 3], torch.zeros_like(edge_output[:, 3]))
    edge_output.sum().backward()
    assert nodes.grad is not None and torch.isfinite(nodes.grad).all()


def test_relation_reranker_consumes_nodes_geometry_and_relation_embeddings() -> None:
    torch.manual_seed(3)
    reranker = RelationAwareReranker(embedding_dim=8, geometry_dim=12)
    base = torch.tensor([0.1, 0.8, 0.2])
    states = torch.randn(3, 8, requires_grad=True)
    edges = (
        EdgeRecord(1, 0, RelationType.PROXIMITY, RelationDirection.LEFT, (0.1,) * 12),
        EdgeRecord(2, 0, RelationType.PROXIMITY, RelationDirection.RIGHT, (0.7,) * 12),
    )

    output = reranker(base, states, edges)

    geometry = torch.tensor([[0.1] * 12, [0.7] * 12])
    destination = states[0].unsqueeze(0).expand(2, -1)
    sources = states[torch.tensor([1, 2])]
    relation_state = reranker.relation_embedding.weight[1].unsqueeze(0).expand(2, -1)
    compatibility = reranker.compatibility[1](
        torch.cat((destination, sources, geometry, relation_state), dim=-1)
    ).squeeze(-1)
    neighbor_gates = torch.softmax(compatibility, dim=0)
    relation_weights = torch.softmax(reranker.relation_weights, dim=0)
    expected_destination = (
        base[0] + 0.30 * relation_weights[1] * (neighbor_gates * base[torch.tensor([1, 2])]).sum()
    )

    assert torch.isfinite(output).all()
    assert torch.allclose(output[0], expected_destination)
    output.sum().backward()
    assert states.grad is not None and float(states.grad.abs().sum()) > 0.0
    assert reranker.relation_embedding.weight.grad is not None
    assert reranker.compatibility[1][0].weight.grad is not None


def test_requested_actionability_preference_and_padding_mask() -> None:
    nodes = (
        NodeRecord(0, NodeType.CONTROL, (0.0, 0.0, 0.2, 0.2), actionability_logits=(-5, 5, 0, 0)),
        NodeRecord(1, NodeType.CONTROL, (0.3, 0.0, 0.5, 0.2), actionability_logits=(5, -5, 0, 0)),
    )
    result = select_top_k_actionable(
        torch.tensor([10.0, 1.0]),
        nodes,
        k=4,
        requested_action_index=0,
    )
    assert result.node_ids == (1, 0, -1, -1)
    assert result.valid.tolist() == [True, True, False, False]


def test_retention_gumbel_masks_and_survival_budget_losses_have_gradients() -> None:
    torch.manual_seed(5)
    scorer = RetentionScorer(8)
    states = torch.randn(2, 4, 8, requires_grad=True)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    mandatory = torch.tensor([[True, False, False, False], [True, False, False, False]])
    output = scorer(states, valid, stochastic=True, mandatory_mask=mandatory)
    losses = retention_loss(
        output,
        target_mask=torch.tensor([[False, True, False, False], [False, True, False, False]]),
        reference_mask=torch.tensor([[False, False, True, False], [False, False, False, False]]),
        token_costs=torch.full((2, 4), 80.0),
        budgets=torch.tensor([120.0, 120.0]),
    )

    assert output.selection_mask[0, 0] == 1.0
    assert output.selection_mask[0, 3] == 0.0
    assert losses.target_count == 2
    assert losses.reference_count == 1
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert scorer.projection.weight.grad is not None
    assert states.grad is not None


class FakeBertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embedding = nn.Embedding(32, 16)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert return_dict and attention_mask.shape == input_ids.shape
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeTokenizer:
    def __call__(self, commands: list[str], **_: object) -> dict[str, torch.Tensor]:
        width = max(len(command.split()) for command in commands) + 1
        ids = torch.zeros((len(commands), width), dtype=torch.long)
        for row, command in enumerate(commands):
            values = [1, *(2 + index for index, _ in enumerate(command.split()))]
            ids[row, : len(values)] = torch.tensor(values)
        return {"input_ids": ids, "attention_mask": ids != 0}


def test_compact_bert_adapter_masks_cls_and_freezing_without_network() -> None:
    model = FakeBertModel()
    adapter = CompactBertAdapter(
        model,
        tokenizer=FakeTokenizer(),
        expected_dimension=16,
    )
    ids, mask = adapter.tokenize(("tap red", "scroll down"))
    output = adapter(ids, mask)

    assert output.token_states.shape == (2, 3, 16)
    assert torch.equal(output.pooled, output.token_states[:, 0])
    assert adapter.wordpiece_embeddings is model.embedding
    adapter.set_trainable(False)
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_mobilevit_adapter_exact_shape_probe_and_gradient() -> None:
    backbone = nn.Sequential(nn.Conv2d(3, 24, 3, stride=4, padding=1), nn.GELU())
    adapter = MobileVitSCropEncoder(backbone, backbone_dimension=24)
    crops = torch.randn(2, 3, 192, 192, requires_grad=True)

    output = adapter(crops)

    assert output.shape == (2, 144, 256)
    assert adapter.shape_probe(batch_size=1)["output_shape"] == [1, 144, 256]
    output.mean().backward()
    assert crops.grad is not None


def test_absolute_box_encoding_is_deterministic_and_position_sensitive() -> None:
    boxes = torch.tensor([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]])
    first = sinusoidal_box_encoding(boxes, 16)
    second = sinusoidal_box_encoding(boxes.clone(), 16)

    assert first.shape == (1, 2, 16)
    assert torch.equal(first, second)
    assert not torch.equal(first[:, 0], first[:, 1])


def _grounding_supervision(command_count: int) -> GroundingSupervision:
    return GroundingSupervision(
        candidate_indices=torch.tensor([0, 1, 0, 1]),
        candidate_mask=torch.ones(command_count, dtype=torch.bool),
        point_local=torch.full((command_count, 2), 0.5),
        point_mask=torch.ones(command_count, dtype=torch.bool),
        action_types=torch.tensor([0, 1, 2, 3]),
        action_mask=torch.ones(command_count, dtype=torch.bool),
        long_press_point_local=torch.full((command_count, 2), 0.4),
        long_press_mask=torch.ones(command_count, dtype=torch.bool),
        scroll_container_indices=torch.tensor([0, 0, 1, 0]),
        scroll_mask=torch.ones(command_count, dtype=torch.bool),
        scroll_delta=torch.zeros(command_count, 2),
        drag_source_indices=torch.tensor([0, 0, 0, 0]),
        drag_destination_indices=torch.tensor([1, 1, 1, 1]),
        drag_mask=torch.ones(command_count, dtype=torch.bool),
        drag_source_point_local=torch.full((command_count, 2), 0.2),
        drag_destination_point_local=torch.full((command_count, 2), 0.8),
        drag_duration=torch.full((command_count, 1), 0.3),
        drag_duration_mask=torch.ones(command_count, dtype=torch.bool),
        target_boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]] * command_count),
        target_box_mask=torch.ones(command_count, dtype=torch.bool),
    )


def test_sparse_grounder_masks_crop_tokens_and_action_specific_losses() -> None:
    torch.manual_seed(8)
    grounder = SparseCandidateGrounder(embedding_dim=16, heads=4, blocks=2)
    command = torch.randn(4, 5, 16)
    crops = torch.randn(4, 3, 4, 16)
    nodes = torch.randn(4, 3, 16)
    candidate_mask = torch.tensor([[True, True, False]] * 4)
    crop_mask = candidate_mask.unsqueeze(-1).expand(-1, -1, 4).clone()
    crop_mask[:, 1, 2:] = False
    boxes = torch.rand(4, 3, 4)
    boxes = torch.cat(
        (
            torch.minimum(boxes[..., :2], boxes[..., 2:]),
            torch.maximum(boxes[..., :2], boxes[..., 2:]),
        ),
        dim=-1,
    )

    output = grounder(
        command,
        crops,
        nodes,
        candidate_mask=candidate_mask,
        crop_mask=crop_mask,
        candidate_boxes=boxes,
    )
    losses = grounding_loss(
        output,
        _grounding_supervision(4),
        candidate_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0]] * 3] * 4),
        confidence_config=ConfidenceReconstructionConfig(warmup_steps=0),
    )

    assert output.candidate_logits[:, 2].max() < -1e8
    assert output.drag_source_logits[:, 2].max() < -1e8
    assert output.drag_destination_logits[:, 2].max() < -1e8
    assert torch.isfinite(losses.total)
    assert losses.confidence_count == 4
    losses.total.backward()
    assert grounder.drag_destination_point_head.weight.grad is not None
    assert grounder.scroll_delta_head.weight.grad is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_masked_action_losses_and_padding_are_finite(dtype: torch.dtype) -> None:
    grounder = SparseCandidateGrounder(embedding_dim=16, heads=4, blocks=1).to(dtype=dtype)
    output = grounder(
        torch.randn(4, 3, 16, dtype=dtype),
        torch.randn(4, 3, 4, 16, dtype=dtype),
        torch.randn(4, 3, 16, dtype=dtype),
        candidate_mask=torch.tensor([[True, True, False]] * 4),
    )
    supervision = replace(_grounding_supervision(4), action_mask=torch.zeros(4, dtype=torch.bool))
    losses = grounding_loss(output, supervision, candidate_boxes=torch.zeros(4, 3, 4))
    assert torch.isfinite(losses.total)
    assert losses.action == losses.point == losses.parameters == 0
    assert bool((output.drag_destination_logits.argmax(-1) != 2).all())
    losses.total.backward()
    assert grounder.candidate_head.weight.grad is not None
    assert torch.isfinite(grounder.candidate_head.weight.grad).all()


def test_missing_target_does_not_train_an_arbitrary_candidate_action() -> None:
    grounder = SparseCandidateGrounder(embedding_dim=16, heads=4, blocks=1)
    output = grounder(torch.randn(4, 3, 16), torch.randn(4, 2, 4, 16), torch.randn(4, 2, 16))
    supervision = replace(
        _grounding_supervision(4), candidate_mask=torch.zeros(4, dtype=torch.bool)
    )
    losses = grounding_loss(output, supervision, candidate_boxes=torch.zeros(4, 2, 4))
    assert losses.action == losses.point == 0
