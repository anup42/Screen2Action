"""End-to-end CPU oracle/cached Screen2Action runtime."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import cast

import torch

from screen2action.data.schema import (
    ActionType,
    ConfidenceOutput,
    EdgeRecord,
    GroundingOutput,
    NodeRecord,
    Point,
)
from screen2action.data.tokenizer import VocabularyTokenizer
from screen2action.models.command_encoder import CommandEncoder
from screen2action.models.crop_encoder import CropTokenEncoder
from screen2action.models.node_encoder import NodeEncoder, node_batch_from_records
from screen2action.models.relation_gat import RelationGraphEncoder, edge_tensors
from screen2action.models.retriever import (
    CosineRetriever,
    RelationAwareReranker,
    RetrievalResult,
    select_top_k_actionable,
)
from screen2action.models.sparse_grounder import SparseCandidateGrounder
from screen2action.perception.oracle import OracleFrame
from screen2action.runtime.cropper import batch_crop_tensor, crop_to_screen_point
from screen2action.runtime.structured_logging import log_event
from screen2action.ssb.codec import SsbCodec, SsbEncoding
from screen2action.ssb.hierarchy import Hierarchy, build_hierarchy
from screen2action.ssb.relations import build_ordinal_edges, build_proximity_edges
from screen2action.ssb.selector import SelectionResult, budget_select
from screen2action.ssb.validation import validate_encoding, validate_graph


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Shared-interface CPU and paper-reference dimensions."""

    embedding_dim: int = 64
    command_layers: int = 2
    graph_layers: int = 1
    command_heads: int = 4
    command_ffn: int = 256
    command_max_length: int = 64
    top_k: int = 4
    ssb_budget: int = 256
    crop_size: int = 64
    crop_tokens: int = 16
    actionability_threshold: float = 0.5
    crop_margin_fraction: float = 0.12
    icon_class_count: int = 87
    visual_feature_dim: int = 16
    relation_lambda: float = 0.30

    @classmethod
    def tiny_cpu(cls) -> PipelineConfig:
        """Return the small CPU architecture profile."""

        return cls()

    @classmethod
    def paper_reference(cls) -> PipelineConfig:
        """Return the paper-reference dimensions without claiming pretrained weights."""

        return cls(
            embedding_dim=256,
            command_layers=6,
            graph_layers=2,
            command_heads=4,
            command_ffn=1024,
            top_k=8,
            ssb_budget=512,
            crop_size=192,
            crop_tokens=144,
        )


@dataclass(frozen=True, slots=True)
class FrameState:
    """Command-independent cached state for one screenshot."""

    screenshot: torch.Tensor
    nodes: tuple[NodeRecord, ...]
    edges: tuple[EdgeRecord, ...]
    hierarchy: Hierarchy
    selection: SelectionResult
    ssb: SsbEncoding
    node_embeddings: torch.Tensor
    edge_index: torch.Tensor
    relation_ids: torch.Tensor
    relative_geometry: torch.Tensor


class Screen2ActionPipeline:
    """Prepare a frame once, then ground multiple commands against its cache."""

    def __init__(
        self,
        *,
        tokenizer: VocabularyTokenizer | None = None,
        config: PipelineConfig | None = None,
        device: torch.device | str = "cpu",
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or PipelineConfig.tiny_cpu()
        self.device = torch.device(device)
        self.logger = logger or logging.getLogger("screen2action.pipeline")
        self.tokenizer = tokenizer or VocabularyTokenizer.default()
        vocabulary_size = max(self.tokenizer.vocab_size, 8)
        self.node_encoder = NodeEncoder(
            embedding_dim=self.config.embedding_dim,
            vocab_size=vocabulary_size,
            icon_class_count=self.config.icon_class_count,
            visual_feature_dim=self.config.visual_feature_dim,
        ).to(self.device)
        self.graph_encoder = RelationGraphEncoder(
            layers=self.config.graph_layers,
            embedding_dim=self.config.embedding_dim,
            heads=self.config.command_heads,
            geometry_dim=12,
        ).to(self.device)
        self.command_encoder = CommandEncoder(
            vocab_size=vocabulary_size,
            embedding_dim=self.config.embedding_dim,
            layers=self.config.command_layers,
            heads=self.config.command_heads,
            feedforward_dim=self.config.command_ffn,
            max_length=self.config.command_max_length,
            dropout=0.0,
        ).to(self.device)
        self.retriever = CosineRetriever(self.config.embedding_dim, self.config.embedding_dim).to(
            self.device
        )
        self.reranker = RelationAwareReranker(
            self.config.embedding_dim,
            lambda_ref=self.config.relation_lambda,
        ).to(self.device)
        self.crop_encoder = CropTokenEncoder(
            embedding_dim=self.config.embedding_dim,
            token_count=self.config.crop_tokens,
            tiny=self.config.embedding_dim < 256,
        ).to(self.device)
        self.grounder = SparseCandidateGrounder(
            embedding_dim=self.config.embedding_dim,
            heads=self.config.command_heads,
            blocks=2,
        ).to(self.device)
        for module in self.modules:
            module.eval()

    @property
    def modules(self) -> tuple[torch.nn.Module, ...]:
        """Return all runtime modules for device/state management."""

        return (
            self.node_encoder,
            self.graph_encoder,
            self.command_encoder,
            self.retriever,
            self.reranker,
            self.crop_encoder,
            self.grounder,
        )

    def _position_edges(
        self, edges: Iterable[EdgeRecord], node_order: tuple[int, ...]
    ) -> tuple[EdgeRecord, ...]:
        positions = {node_id: index for index, node_id in enumerate(node_order)}
        return tuple(
            replace(edge, src=positions[edge.src], dst=positions[edge.dst])
            for edge in edges
            if edge.src in positions and edge.dst in positions
        )

    def prepare_frame(self, screenshot: torch.Tensor, nodes: Iterable[NodeRecord]) -> FrameState:
        """Construct graph, SSB, selection, and cached node embeddings once."""

        if screenshot.ndim != 3 or screenshot.shape[0] != 3:
            raise ValueError("screenshot must have shape [3, H, W]")
        hierarchy = build_hierarchy(list(nodes))
        containment = hierarchy.containment_edges
        proximity = build_proximity_edges(
            hierarchy.nodes,
            parent_by_child=hierarchy.parent_by_child,
        )
        ordinal = build_ordinal_edges(
            hierarchy.nodes,
            parent_by_child=hierarchy.parent_by_child,
        )
        graph_edges = tuple(containment + proximity + ordinal)
        validate_graph(hierarchy.nodes, graph_edges)
        ordered_nodes = tuple(
            next(node for node in hierarchy.nodes if node.node_id == node_id)
            for node_id in hierarchy.depth_first_order
        )
        selection = budget_select(
            ordered_nodes,
            graph_edges,
            budget=self.config.ssb_budget,
            node_order=hierarchy.depth_first_order,
        )
        selected_ids = set(selection.selected_node_ids)
        selected_order = tuple(
            node_id for node_id in hierarchy.depth_first_order if node_id in selected_ids
        )
        ssb = SsbCodec().encode(
            hierarchy.nodes,
            graph_edges,
            selected_node_ids=selected_order,
            node_order=hierarchy.depth_first_order,
        )
        validate_encoding(ssb)
        if ssb.total_cost > self.config.ssb_budget:
            raise ValueError("serialized SSB exceeds configured budget")
        node_batch = node_batch_from_records(
            hierarchy.nodes,
            text_length=16,
            icon_class_count=self.config.icon_class_count,
            visual_feature_dim=self.config.visual_feature_dim,
            device=self.device,
        )
        position_edges = self._position_edges(
            graph_edges, tuple(node.node_id for node in hierarchy.nodes)
        )
        edge_index, relation_ids, geometry = edge_tensors(
            position_edges,
            device=self.device,
        )
        with torch.no_grad():
            embeddings = self.node_encoder(node_batch)
            embeddings = self.graph_encoder(embeddings, edge_index, relation_ids, geometry)
        node_positions = {
            node_id: index for index, node_id in enumerate(node.node_id for node in hierarchy.nodes)
        }
        selected_positions = [node_positions[node_id] for node_id in selected_order]
        state = FrameState(
            screenshot=screenshot.to(self.device),
            nodes=hierarchy.nodes,
            edges=graph_edges,
            hierarchy=hierarchy,
            selection=selection,
            ssb=ssb,
            node_embeddings=embeddings[selected_positions],
            edge_index=edge_index,
            relation_ids=relation_ids,
            relative_geometry=geometry,
        )
        log_event(
            self.logger,
            "frame_prepared",
            level=logging.DEBUG,
            fields={
                "node_count": len(state.nodes),
                "edge_count": len(state.edges),
                "selected_node_count": len(state.ssb.ordered_node_ids),
                "ssb_token_count": state.ssb.total_cost,
                "ssb_budget": self.config.ssb_budget,
                "height": int(state.screenshot.shape[-2]),
                "width": int(state.screenshot.shape[-1]),
            },
        )
        return state

    def prepare_oracle_frame(self, oracle_frame: OracleFrame) -> FrameState:
        """Convenience wrapper for the CPU oracle perception output."""

        return self.prepare_frame(oracle_frame.screenshot, oracle_frame.nodes)

    def _selected_edges(self, state: FrameState) -> tuple[EdgeRecord, ...]:
        selected_positions = {
            node_id: index for index, node_id in enumerate(state.ssb.ordered_node_ids)
        }
        return tuple(
            replace(edge, src=selected_positions[edge.src], dst=selected_positions[edge.dst])
            for edge in state.edges
            if edge.src in selected_positions and edge.dst in selected_positions
        )

    def ground(self, state: FrameState, command: str) -> GroundingOutput:
        """Retrieve fixed-K candidates, crop them, and predict a screen point."""

        if not command.strip():
            raise ValueError("command cannot be empty")
        input_ids = torch.tensor(
            [self.tokenizer.encode(command, self.config.command_max_length)],
            dtype=torch.long,
            device=self.device,
        )
        command_mask = input_ids != self.tokenizer.pad_id
        with torch.no_grad():
            command_encoding = self.command_encoder(input_ids, command_mask)
            base_scores = self.retriever(command_encoding.pooled[0], state.node_embeddings)
            reranked = self.reranker(
                base_scores, state.node_embeddings, self._selected_edges(state)
            )
            selected_nodes = tuple(
                next(node for node in state.nodes if node.node_id == node_id)
                for node_id in state.ssb.ordered_node_ids
            )
            retrieval: RetrievalResult = select_top_k_actionable(
                reranked,
                selected_nodes,
                k=self.config.top_k,
                actionability_threshold=self.config.actionability_threshold,
            )
            boxes = tuple(
                selected_nodes[int(index)].box_xyxy_norm for index in retrieval.node_indices
            )
            crops, crop_specs = batch_crop_tensor(
                state.screenshot.detach().cpu(),
                list(boxes),
                output_size=self.config.crop_size,
                margin_fraction=self.config.crop_margin_fraction,
            )
            crop_tokens = self.crop_encoder(crops.to(self.device)).reshape(
                1, self.config.top_k, self.config.crop_tokens, self.config.embedding_dim
            )
            node_tokens = state.node_embeddings[retrieval.node_indices].unsqueeze(0)
            grounded = self.grounder(
                command_encoding.token_states,
                crop_tokens,
                node_tokens,
                command_mask=command_mask,
                candidate_mask=retrieval.valid.unsqueeze(0),
            )
            candidate_index = int(torch.argmax(grounded.candidate_logits[0]).item())
            selected_node_index = int(retrieval.node_indices[candidate_index].item())
            point_local = cast(
                Point,
                tuple(float(value) for value in grounded.point_local[0, candidate_index]),
            )
            point_screen = crop_to_screen_point(
                point_local, crop_specs[candidate_index].expanded_box_xyxy_norm
            )
            action_index = int(torch.argmax(grounded.action_type_logits[0, candidate_index]).item())
            action_type = (
                ActionType.CLICK,
                ActionType.DRAG,
                ActionType.SCROLL,
                ActionType.LONG_PRESS,
            )[action_index]
            confidence_logit = float(grounded.confidence_logits[0, candidate_index])
            confidence_probability = float(
                torch.sigmoid(grounded.confidence_logits[0, candidate_index])
            )
        log_event(
            self.logger,
            "command_grounded",
            level=logging.DEBUG,
            fields={
                "candidate_count": len(retrieval.candidates),
                "candidate_index": candidate_index,
                "node_id": selected_nodes[selected_node_index].node_id,
                "action_type": action_type.value,
            },
        )
        return GroundingOutput(
            node_id=selected_nodes[selected_node_index].node_id,
            point_xy_norm=point_screen,
            action_type=action_type,
            action_parameters={
                "p0": float(grounded.action_parameter[0, candidate_index, 0]),
                "p1": float(grounded.action_parameter[0, candidate_index, 1]),
                "p2": float(grounded.action_parameter[0, candidate_index, 2]),
                "p3": float(grounded.action_parameter[0, candidate_index, 3]),
            },
            candidate_logits=tuple(float(value) for value in grounded.candidate_logits[0]),
            confidence=ConfidenceOutput(confidence_logit, confidence_probability),
        )
