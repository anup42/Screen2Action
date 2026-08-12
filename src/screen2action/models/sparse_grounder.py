"""Genuinely grouped candidate-conditioned cross-attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class GroundingTensorOutput:
    """Tensor predictions from candidate-conditioned grounding."""

    candidate_logits: torch.Tensor
    point_local: torch.Tensor
    action_type_logits: torch.Tensor
    action_parameter: torch.Tensor
    long_press_point_local: torch.Tensor
    scroll_container_logits: torch.Tensor
    scroll_delta: torch.Tensor
    drag_source_logits: torch.Tensor
    drag_destination_logits: torch.Tensor
    drag_source_point_local: torch.Tensor
    drag_destination_point_local: torch.Tensor
    drag_duration: torch.Tensor
    confidence_logits: torch.Tensor
    command_states: torch.Tensor
    candidate_states: torch.Tensor


def sinusoidal_box_encoding(boxes: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Encode normalized absolute `xyxy` boxes as deterministic 2-D sinusoidal keys."""

    if boxes.shape[-1] != 4 or embedding_dim <= 0 or embedding_dim % 8 != 0:
        raise ValueError("boxes must end in four values and embedding_dim must divide by eight")
    frequency_count = embedding_dim // 8
    denominator = max(frequency_count - 1, 1)
    frequencies = torch.exp(
        torch.arange(frequency_count, device=boxes.device, dtype=boxes.dtype)
        * (-math.log(10_000.0) / denominator)
    )
    angles = boxes.unsqueeze(-1) * frequencies * (2.0 * math.pi)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)


class SparseCandidateGrounder(nn.Module):
    """Two-stream sparse attention with one isolated batch group per candidate.

    Command tokens attend to the flattened candidate groups. Candidate groups
    are then processed as independent `[batch * K]` items whose key/value
    context contains only the shared command stream and that candidate's own
    node/crop tokens. No dense cross-candidate attention matrix is built.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        heads: int = 4,
        blocks: int = 2,
        action_count: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % heads != 0 or min(embedding_dim, heads, blocks) <= 0:
            raise ValueError("invalid sparse-grounder dimensions")
        self.embedding_dim = embedding_dim
        self.blocks = blocks
        self.command_attn = nn.ModuleList(
            nn.MultiheadAttention(
                embedding_dim,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(blocks)
        )
        self.local_attn = nn.ModuleList(
            nn.MultiheadAttention(
                embedding_dim,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(blocks)
        )
        self.command_norm = nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(blocks))
        self.candidate_norm = nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(blocks))
        self.command_output = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim) for _ in range(blocks)
        )
        self.candidate_output = nn.ModuleList(
            nn.Linear(embedding_dim, embedding_dim) for _ in range(blocks)
        )
        self.candidate_head = nn.Linear(embedding_dim, 1)
        self.point_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 2)
        )
        self.action_head = nn.Linear(embedding_dim, action_count)
        self.action_parameter_head = nn.Linear(embedding_dim, 4)
        self.long_press_point_head = nn.Linear(embedding_dim, 2)
        self.scroll_container_head = nn.Linear(embedding_dim, 1)
        self.scroll_delta_head = nn.Linear(embedding_dim, 2)
        self.drag_source_head = nn.Linear(embedding_dim, 1)
        self.drag_destination_head = nn.Linear(embedding_dim, 1)
        self.drag_source_point_head = nn.Linear(embedding_dim, 2)
        self.drag_destination_point_head = nn.Linear(embedding_dim, 2)
        self.drag_duration_head = nn.Linear(embedding_dim, 1)
        self.confidence_head = nn.Linear(embedding_dim, 1)

    def _command_update(
        self,
        command_states: torch.Tensor,
        group_states: torch.Tensor,
        command_mask: torch.Tensor,
        group_mask: torch.Tensor,
        group_position: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        batch, candidates, group_length, embedding = group_states.shape
        flattened = group_states.reshape(batch, candidates * group_length, embedding)
        flattened_position = group_position.reshape(batch, candidates * group_length, embedding)
        key_padding = ~group_mask.reshape(batch, candidates * group_length).bool()
        flattened_values = flattened
        all_masked = key_padding.all(dim=1)
        if bool(all_masked.any()):
            key_padding = key_padding.clone()
            flattened_values = flattened.clone()
            flattened_position = flattened_position.clone()
            key_padding[all_masked, 0] = False
            flattened_values[all_masked, 0] = 0.0
            flattened_position[all_masked, 0] = 0.0
        attended, _ = self.command_attn[block_index](
            command_states,
            flattened_values + flattened_position,
            flattened_values,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        updated = self.command_norm[block_index](
            command_states + self.command_output[block_index](attended)
        )
        return cast(torch.Tensor, updated * command_mask.unsqueeze(-1).to(updated.dtype))

    def local_candidate_update(
        self,
        command_states: torch.Tensor,
        group_states: torch.Tensor,
        command_mask: torch.Tensor,
        group_mask: torch.Tensor | None = None,
        group_position: torch.Tensor | None = None,
        block_index: int = 0,
    ) -> torch.Tensor:
        """Update each group with a fixed shared command stream for isolation tests."""

        batch, candidates, group_length, embedding = group_states.shape
        flattened_groups = group_states.reshape(batch * candidates, group_length, embedding)
        repeated_command = command_states.unsqueeze(1).expand(-1, candidates, -1, -1)
        repeated_command = repeated_command.reshape(
            batch * candidates, command_states.shape[1], embedding
        )
        repeated_mask = (~command_mask.bool()).unsqueeze(1).expand(-1, candidates, -1)
        repeated_mask = repeated_mask.reshape(batch * candidates, command_states.shape[1])
        if group_mask is None:
            group_mask = torch.ones(
                (batch, candidates, group_length), dtype=torch.bool, device=group_states.device
            )
        if group_position is None:
            group_position = torch.zeros_like(group_states)
        if group_mask.shape != (batch, candidates, group_length):
            raise ValueError("group_mask must have shape [B,K,1+P]")
        if group_position.shape != group_states.shape:
            raise ValueError("group_position must match group states")
        flattened_group_mask = ~group_mask.reshape(batch * candidates, group_length).bool()
        local_context = torch.cat((repeated_command, flattened_groups), dim=1)
        local_position = torch.cat(
            (
                torch.zeros_like(repeated_command),
                group_position.reshape(batch * candidates, group_length, embedding),
            ),
            dim=1,
        )
        context_mask = torch.cat(
            (repeated_mask, flattened_group_mask),
            dim=1,
        )
        all_masked = context_mask.all(dim=1)
        if bool(all_masked.any()):
            context_mask = context_mask.clone()
            local_context = local_context.clone()
            local_position = local_position.clone()
            context_mask[all_masked, 0] = False
            local_context[all_masked, 0] = 0.0
            local_position[all_masked, 0] = 0.0
        attended, _ = self.local_attn[block_index](
            flattened_groups,
            local_context + local_position,
            local_context,
            key_padding_mask=context_mask,
            need_weights=False,
        )
        updated = self.candidate_norm[block_index](
            flattened_groups + self.candidate_output[block_index](attended)
        )
        updated = updated.reshape(batch, candidates, group_length, embedding)
        return cast(torch.Tensor, updated * group_mask.unsqueeze(-1).to(updated.dtype))

    def forward(
        self,
        command_states: torch.Tensor,
        candidate_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        *,
        command_mask: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        crop_mask: torch.Tensor | None = None,
        candidate_boxes: torch.Tensor | None = None,
    ) -> GroundingTensorOutput:
        """Run two sparse conditioning blocks and predict click/action outputs."""

        if command_states.ndim != 3 or candidate_tokens.ndim != 4 or node_tokens.ndim != 3:
            raise ValueError("command/candidate/node tensors have invalid ranks")
        batch, candidates, _, embedding = candidate_tokens.shape
        if node_tokens.shape != (batch, candidates, embedding):
            raise ValueError("node_tokens must have shape [B, K, D]")
        if command_mask is None:
            command_mask = torch.ones(
                command_states.shape[:2], dtype=torch.bool, device=command_states.device
            )
        if candidate_mask is None:
            candidate_mask = torch.ones(
                (batch, candidates), dtype=torch.bool, device=command_states.device
            )
        if candidate_mask.shape != (batch, candidates):
            raise ValueError("candidate_mask must have shape [B,K]")
        crop_count = candidate_tokens.shape[2]
        if crop_mask is None:
            crop_mask = candidate_mask.unsqueeze(-1).expand(-1, -1, crop_count)
        if crop_mask.shape != (batch, candidates, crop_count):
            raise ValueError("crop_mask must have shape [B,K,P]")
        crop_mask = crop_mask.bool() & candidate_mask.bool().unsqueeze(-1)
        if candidate_boxes is None:
            candidate_boxes = torch.zeros(
                (batch, candidates, 4),
                dtype=command_states.dtype,
                device=command_states.device,
            )
        if candidate_boxes.shape != (batch, candidates, 4):
            raise ValueError("candidate_boxes must have shape [B,K,4]")
        group_states = torch.cat((node_tokens.unsqueeze(2), candidate_tokens), dim=2)
        group_mask = torch.cat((candidate_mask.bool().unsqueeze(-1), crop_mask), dim=-1)
        box_position = sinusoidal_box_encoding(candidate_boxes, self.embedding_dim)
        group_position = box_position.unsqueeze(2).expand(-1, -1, crop_count + 1, -1)
        for block_index in range(self.blocks):
            command_states = self._command_update(
                command_states,
                group_states,
                command_mask,
                group_mask,
                group_position,
                block_index,
            )
            group_states = self.local_candidate_update(
                command_states,
                group_states,
                command_mask,
                group_mask,
                group_position,
                block_index,
            )
        candidate_states = group_states[:, :, 0]
        candidate_logits = self.candidate_head(candidate_states).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask.bool(), -1e9)
        point_local = torch.sigmoid(self.point_head(candidate_states))
        action_type_logits = self.action_head(candidate_states)
        action_parameter = torch.tanh(self.action_parameter_head(candidate_states))
        long_press_point = torch.sigmoid(self.long_press_point_head(candidate_states))
        scroll_container_logits = self.scroll_container_head(candidate_states).squeeze(-1)
        scroll_container_logits = scroll_container_logits.masked_fill(~candidate_mask.bool(), -1e9)
        scroll_delta = torch.tanh(self.scroll_delta_head(candidate_states))
        drag_source_logits = self.drag_source_head(candidate_states).squeeze(-1)
        drag_destination_logits = self.drag_destination_head(candidate_states).squeeze(-1)
        drag_source_point = torch.sigmoid(self.drag_source_point_head(candidate_states))
        drag_destination_point = torch.sigmoid(self.drag_destination_point_head(candidate_states))
        drag_duration = torch.sigmoid(self.drag_duration_head(candidate_states))
        confidence_logits = self.confidence_head(candidate_states).squeeze(-1)
        return GroundingTensorOutput(
            candidate_logits=candidate_logits,
            point_local=point_local,
            action_type_logits=action_type_logits,
            action_parameter=action_parameter,
            long_press_point_local=long_press_point,
            scroll_container_logits=scroll_container_logits,
            scroll_delta=scroll_delta,
            drag_source_logits=drag_source_logits,
            drag_destination_logits=drag_destination_logits,
            drag_source_point_local=drag_source_point,
            drag_destination_point_local=drag_destination_point,
            drag_duration=drag_duration,
            confidence_logits=confidence_logits,
            command_states=command_states,
            candidate_states=candidate_states,
        )
