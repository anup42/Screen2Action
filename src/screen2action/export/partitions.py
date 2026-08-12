"""Fixed-tensor wrappers for the four logical Screen2Action partitions."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from screen2action.models.command_encoder import CommandEncoding
from screen2action.models.node_encoder import NodeBatch, NodeEncoder
from screen2action.models.relation_gat import RelationGraphEncoder
from screen2action.models.retention import RetentionScorer
from screen2action.models.retriever import CosineRetriever, RelationAwareReranker
from screen2action.models.sparse_grounder import SparseCandidateGrounder
from screen2action.perception.icon_actionability import MobileNetV3IconActionability


class VisualEncoderPartition(nn.Module):
    """Icon, independent actionability, and visual-node feature partition."""

    def __init__(self, visual: MobileNetV3IconActionability) -> None:
        super().__init__()
        self.visual = visual

    def forward(
        self, crops: torch.Tensor, valid_crops: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.visual.forward_export(crops, valid_crops)
        return output.icon_logits, output.actionability_logits, output.visual_features


class GraphRetentionPartition(nn.Module):
    """Multimodal node fusion, relation graph encoding, and retention scores."""

    def __init__(
        self,
        node_encoder: NodeEncoder,
        graph_encoder: RelationGraphEncoder,
        retention_scorer: RetentionScorer,
    ) -> None:
        super().__init__()
        self.node_encoder = node_encoder
        self.graph_encoder = graph_encoder
        self.retention_scorer = retention_scorer

    def forward(
        self,
        node_type_ids: torch.Tensor,
        original_class_ids: torch.Tensor,
        original_class_mask: torch.Tensor,
        boxes_xyxy_norm: torch.Tensor,
        hierarchy_depth: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        icon_probabilities: torch.Tensor,
        icon_mask: torch.Tensor,
        visual_features: torch.Tensor,
        visual_mask: torch.Tensor,
        actionability_logits: torch.Tensor,
        actionability_mask: torch.Tensor,
        confidences: torch.Tensor,
        valid_nodes: torch.Tensor,
        relation_mask: torch.Tensor,
        relative_geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes = NodeBatch(
            node_type_ids=node_type_ids,
            original_class_ids=original_class_ids,
            original_class_mask=original_class_mask,
            boxes=boxes_xyxy_norm,
            hierarchy_depth=hierarchy_depth,
            text_ids=text_ids,
            text_mask=text_mask,
            icon_probabilities=icon_probabilities,
            icon_mask=icon_mask,
            visual_features=visual_features,
            visual_mask=visual_mask,
            actionability_logits=actionability_logits,
            actionability_mask=actionability_mask,
            confidences=confidences,
            valid_mask=valid_nodes,
        )
        node_states = self.node_encoder(nodes) * valid_nodes.unsqueeze(-1).to(visual_features.dtype)
        graph_states = self.graph_encoder.forward_export_dense(
            node_states,
            relation_mask,
            relative_geometry,
            valid_nodes,
        )
        retention = self.retention_scorer(graph_states, valid_nodes, stochastic=False)
        return node_states, graph_states, retention.logits, retention.scores


class CommandRetrievalPartition(nn.Module):
    """Command encoding, cosine retrieval, and fixed-grid relation reranking."""

    def __init__(
        self,
        command_encoder: nn.Module,
        retriever: CosineRetriever,
        reranker: RelationAwareReranker,
    ) -> None:
        super().__init__()
        self.command_encoder = command_encoder
        self.retriever = retriever
        self.reranker = reranker

    def forward(
        self,
        command_ids: torch.Tensor,
        command_mask: torch.Tensor,
        graph_states: torch.Tensor,
        relation_mask: torch.Tensor,
        relative_geometry: torch.Tensor,
        valid_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        command = cast(CommandEncoding, self.command_encoder(command_ids, command_mask))
        query = F.normalize(self.retriever.query_projection(command.pooled), dim=-1)
        nodes = F.normalize(self.retriever.node_projection(graph_states), dim=-1)
        base_scores = (nodes * query.unsqueeze(1)).sum(dim=-1)
        base_scores = base_scores.masked_fill(~valid_nodes.bool(), -1e9)
        reranked = self.reranker.forward_dense(
            base_scores,
            graph_states,
            relation_mask,
            relative_geometry,
            valid_nodes,
        )
        return command.token_states, command.pooled, base_scores, reranked


class CropGroundingPartition(nn.Module):
    """Candidate crop encoder, sparse grounder, action heads, and confidence."""

    def __init__(self, crop_encoder: nn.Module, grounder: SparseCandidateGrounder) -> None:
        super().__init__()
        self.crop_encoder = crop_encoder
        self.grounder = grounder

    def forward(
        self,
        candidate_crops: torch.Tensor,
        command_states: torch.Tensor,
        command_mask: torch.Tensor,
        node_tokens: torch.Tensor,
        candidate_mask: torch.Tensor,
        crop_mask: torch.Tensor,
        candidate_boxes_xyxy_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch, candidates = candidate_crops.shape[:2]
        flat_crops = candidate_crops.reshape(
            batch * candidates,
            candidate_crops.shape[2],
            candidate_crops.shape[3],
            candidate_crops.shape[4],
        )
        flat_tokens = cast(torch.Tensor, self.crop_encoder(flat_crops))
        candidate_tokens = flat_tokens.reshape(
            batch,
            candidates,
            flat_tokens.shape[1],
            flat_tokens.shape[2],
        )
        output = self.grounder(
            command_states,
            candidate_tokens,
            node_tokens,
            command_mask=command_mask,
            candidate_mask=candidate_mask,
            crop_mask=crop_mask,
            candidate_boxes=candidate_boxes_xyxy_norm,
        )
        return (
            output.candidate_logits,
            output.point_local,
            output.action_type_logits,
            output.action_parameter,
            output.long_press_point_local,
            output.scroll_container_logits,
            output.scroll_delta,
            output.drag_source_logits,
            output.drag_destination_logits,
            output.drag_source_point_local,
            output.drag_destination_point_local,
            output.drag_duration,
            output.confidence_logits,
        )


VISUAL_INPUT_NAMES = ("semantic_crops", "valid_crops")
VISUAL_OUTPUT_NAMES = ("icon_logits", "actionability_logits", "visual_features")

GRAPH_INPUT_NAMES = (
    "node_type_ids",
    "original_class_ids",
    "original_class_mask",
    "boxes_xyxy_norm",
    "hierarchy_depth",
    "text_ids",
    "text_mask",
    "icon_probabilities",
    "icon_mask",
    "visual_features",
    "visual_mask",
    "actionability_logits",
    "actionability_mask",
    "confidences",
    "valid_nodes",
    "relation_mask",
    "relative_geometry",
)
GRAPH_OUTPUT_NAMES = ("node_states", "graph_states", "retention_logits", "retention_scores")

COMMAND_INPUT_NAMES = (
    "command_ids",
    "command_mask",
    "graph_states",
    "relation_mask",
    "relative_geometry",
    "valid_nodes",
)
COMMAND_OUTPUT_NAMES = (
    "command_token_states",
    "command_pooled",
    "retrieval_scores",
    "reranked_scores",
)

GROUNDING_INPUT_NAMES = (
    "candidate_crops",
    "command_token_states",
    "command_mask",
    "candidate_node_states",
    "candidate_mask",
    "crop_token_mask",
    "candidate_boxes_xyxy_norm",
)
GROUNDING_OUTPUT_NAMES = (
    "candidate_logits",
    "point_local",
    "action_type_logits",
    "action_parameter",
    "long_press_point_local",
    "scroll_container_logits",
    "scroll_delta",
    "drag_source_logits",
    "drag_destination_logits",
    "drag_source_point_local",
    "drag_destination_point_local",
    "drag_duration",
    "confidence_logits",
)
