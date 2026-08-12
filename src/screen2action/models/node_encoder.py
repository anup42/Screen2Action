"""Multimodal node encoder with explicit learned null embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from screen2action.data.schema import NodeRecord, NodeType

_NODE_TYPE_TO_ID = {node_type: index for index, node_type in enumerate(NodeType)}


@dataclass(frozen=True, slots=True)
class NodeBatch:
    """Tensorized node features for one screen or a padded batch."""

    node_type_ids: torch.Tensor
    boxes: torch.Tensor
    text_ids: torch.Tensor
    text_mask: torch.Tensor
    icon_probabilities: torch.Tensor
    icon_mask: torch.Tensor
    visual_features: torch.Tensor
    visual_mask: torch.Tensor
    actionability_logits: torch.Tensor
    actionability_mask: torch.Tensor
    valid_mask: torch.Tensor

    @property
    def node_count(self) -> int:
        return int(self.node_type_ids.shape[-1])


def node_batch_from_records(
    nodes: tuple[NodeRecord, ...] | list[NodeRecord],
    *,
    text_length: int = 16,
    icon_class_count: int = 87,
    visual_feature_dim: int = 16,
    device: torch.device | str | None = None,
) -> NodeBatch:
    """Convert canonical records to fixed-shape tensors."""

    if text_length <= 0 or icon_class_count <= 0 or visual_feature_dim <= 0:
        raise ValueError("feature dimensions must be positive")
    node_list = list(nodes)
    count = len(node_list)
    target_device = torch.device(device) if device is not None else None
    node_type_ids = torch.tensor(
        [_NODE_TYPE_TO_ID[node.node_type] for node in node_list],
        dtype=torch.long,
        device=target_device,
    )
    boxes = torch.tensor(
        [node.box_xyxy_norm for node in node_list], dtype=torch.float32, device=target_device
    )
    text_ids = torch.zeros((count, text_length), dtype=torch.long, device=target_device)
    text_mask = torch.zeros((count, text_length), dtype=torch.bool, device=target_device)
    icon_probabilities = torch.zeros(
        (count, icon_class_count), dtype=torch.float32, device=target_device
    )
    icon_mask = torch.zeros(count, dtype=torch.bool, device=target_device)
    visual_features = torch.zeros(
        (count, visual_feature_dim), dtype=torch.float32, device=target_device
    )
    visual_mask = torch.zeros(count, dtype=torch.bool, device=target_device)
    actionability_logits = torch.zeros((count, 4), dtype=torch.float32, device=target_device)
    actionability_mask = torch.zeros((count, 4), dtype=torch.bool, device=target_device)
    for index, node in enumerate(node_list):
        text = node.text_token_ids[:text_length]
        if text:
            text_ids[index, : len(text)] = torch.tensor(
                text, dtype=torch.long, device=target_device
            )
            text_mask[index, : len(text)] = True
        if node.icon_probabilities:
            values = node.icon_probabilities[:icon_class_count]
            icon_probabilities[index, : len(values)] = torch.tensor(
                values, dtype=torch.float32, device=target_device
            )
            icon_mask[index] = True
        if node.roi_visual_feature:
            values = node.roi_visual_feature[:visual_feature_dim]
            visual_features[index, : len(values)] = torch.tensor(
                values, dtype=torch.float32, device=target_device
            )
            visual_mask[index] = True
        actionability_logits[index] = torch.tensor(
            node.actionability_logits, dtype=torch.float32, device=target_device
        )
        actionability_mask[index] = torch.tensor(
            node.actionability_mask, dtype=torch.bool, device=target_device
        )
    return NodeBatch(
        node_type_ids=node_type_ids,
        boxes=boxes,
        text_ids=text_ids,
        text_mask=text_mask,
        icon_probabilities=icon_probabilities,
        icon_mask=icon_mask,
        visual_features=visual_features,
        visual_mask=visual_mask,
        actionability_logits=actionability_logits,
        actionability_mask=actionability_mask,
        valid_mask=torch.ones(count, dtype=torch.bool, device=target_device),
    )


class NodeEncoder(nn.Module):
    """Encode text, icon, visual, geometry, and actionability node fields."""

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        vocab_size: int = 16_384,
        icon_class_count: int = 87,
        visual_feature_dim: int = 16,
        type_count: int = len(NodeType),
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or vocab_size <= 0:
            raise ValueError("embedding_dim and vocab_size must be positive")
        self.embedding_dim = embedding_dim
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.type_embedding = nn.Embedding(type_count, embedding_dim)
        self.icon_projection = nn.Linear(icon_class_count, embedding_dim)
        self.visual_projection = nn.Linear(visual_feature_dim, embedding_dim)
        self.geometry_projection = nn.Sequential(nn.Linear(4, embedding_dim), nn.GELU())
        self.actionability_projection = nn.Linear(4, embedding_dim)
        self.null_text = nn.Parameter(torch.zeros(embedding_dim))
        self.null_icon = nn.Parameter(torch.zeros(embedding_dim))
        self.null_visual = nn.Parameter(torch.zeros(embedding_dim))
        self.output = nn.Sequential(
            nn.Linear(embedding_dim * 6, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, batch: NodeBatch) -> torch.Tensor:
        """Return `[N, d]` or `[B,N,d]` node embeddings."""

        text_embeddings = self.text_embedding(batch.text_ids)
        text_mask = batch.text_mask.unsqueeze(-1)
        text_sum = (text_embeddings * text_mask).sum(dim=-2)
        text_count = batch.text_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(text_sum.dtype)
        text = text_sum / text_count
        text = torch.where(batch.text_mask.any(dim=-1, keepdim=True), text, self.null_text)
        icon = self.icon_projection(batch.icon_probabilities)
        icon = torch.where(batch.icon_mask.unsqueeze(-1), icon, self.null_icon)
        visual = self.visual_projection(batch.visual_features)
        visual = torch.where(batch.visual_mask.unsqueeze(-1), visual, self.null_visual)
        type_embedding = self.type_embedding(batch.node_type_ids)
        geometry = self.geometry_projection(batch.boxes)
        actionability = self.actionability_projection(
            batch.actionability_logits
            * batch.actionability_mask.to(batch.actionability_logits.dtype)
        )
        combined = torch.cat((type_embedding, text, icon, visual, geometry, actionability), dim=-1)
        return cast(torch.Tensor, self.output(combined))
