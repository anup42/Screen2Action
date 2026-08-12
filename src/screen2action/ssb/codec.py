"""Versioned integer-token SSB codec used by the CPU milestone.

The paper gives the 512-token budget but not the grammar. This module therefore
implements the explicitly documented reconstruction in `docs/DECISIONS.md`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from screen2action.data.schema import (
    EdgeRecord,
    NodeRecord,
    NodeType,
    RelationAxis,
    RelationDirection,
    RelationType,
)
from screen2action.ssb.geometry import quantize_box, quantize_coordinate

MAGIC_TOKEN = 4096
CODEC_VERSION = 1
END_TOKEN = 4097
NODE_START_TOKEN = 4098
NODE_END_TOKEN = 4099
TEXT_START_TOKEN = 4100
TEXT_END_TOKEN = 4101
RELATION_SLOT_TOKEN = 4102
NULL_REFERENCE = 0
MAX_RELATION_SLOTS = 8
RELATION_SLOT_TOKENS = 5
HEADER_TOKEN_COUNT = 3
FOOTER_TOKEN_COUNT = 1

_NODE_TYPE_CODES = {node_type: index for index, node_type in enumerate(NodeType, start=1)}
_NODE_TYPE_FROM_CODE = {code: node_type for node_type, code in _NODE_TYPE_CODES.items()}
_DIRECTION_CODES = {
    RelationDirection.NONE: 0,
    RelationDirection.LEFT: 1,
    RelationDirection.RIGHT: 2,
    RelationDirection.ABOVE: 3,
    RelationDirection.BELOW: 4,
    RelationDirection.PREVIOUS: 5,
    RelationDirection.NEXT: 6,
}
_DIRECTION_FROM_CODE = {code: direction for direction, code in _DIRECTION_CODES.items()}


def _fixed_node_token_count() -> int:
    # Start, type, four coordinates, depth, action bits, three confidences,
    # icon class, parent reference, text delimiters, relation slots, end.
    return 1 + 1 + 4 + 1 + 1 + 3 + 1 + 1 + 2 + (MAX_RELATION_SLOTS * RELATION_SLOT_TOKENS) + 1


FIXED_NODE_TOKEN_COUNT = _fixed_node_token_count()


@dataclass(frozen=True, slots=True)
class DecodedRelation:
    """One decoded relation slot; a null destination means it was repaired."""

    relation: RelationType | None
    direction: RelationDirection | None
    destination_index: int | None
    geometry_bucket: int


@dataclass(frozen=True, slots=True)
class DecodedNode:
    """Lossless grammar fields needed to validate an encoded node record."""

    node_index: int
    node_type: NodeType
    quantized_box: tuple[int, int, int, int]
    depth: int
    actionability_bits: int
    detector_confidence_bin: int
    ocr_confidence_bin: int
    icon_confidence_bin: int
    icon_class: int
    parent_index: int | None
    text_token_ids: tuple[int, ...]
    relation_slots: tuple[DecodedRelation, ...]


@dataclass(frozen=True, slots=True)
class DecodedSsb:
    """Decoded SSB stream and its graph-local node indices."""

    version: int
    nodes: tuple[DecodedNode, ...]
    tokens_consumed: int


@dataclass(frozen=True, slots=True)
class SsbEncoding:
    """Encoded graph plus the exact costs and ID remapping used to create it."""

    tokens: tuple[int, ...]
    ordered_node_ids: tuple[int, ...]
    node_costs: Mapping[int, int]
    node_indices: Mapping[int, int]
    total_cost: int


def node_token_cost(node: NodeRecord) -> int:
    """Return the exact fixed-grammar cost for one node."""

    return FIXED_NODE_TOKEN_COUNT + len(node.text_token_ids)


def graph_fixed_token_cost() -> int:
    """Return the header/footer cost included in every non-empty graph."""

    return HEADER_TOKEN_COUNT + FOOTER_TOKEN_COUNT


def _icon_class(node: NodeRecord) -> int:
    if not node.icon_probabilities:
        return 0
    return (
        max(range(len(node.icon_probabilities)), key=lambda index: node.icon_probabilities[index])
        + 1
    )


def _actionability_bits(node: NodeRecord) -> int:
    return sum(
        (1 << index)
        for index, (logit, known) in enumerate(
            zip(node.actionability_logits, node.actionability_mask, strict=True)
        )
        if known and logit >= 0.0
    )


def _relation_code(edge: EdgeRecord) -> int:
    if edge.relation is RelationType.PROXIMITY:
        return 1
    if edge.relation is RelationType.ORDINAL:
        return 3 if edge.axis is RelationAxis.COLUMN else 2
    raise ValueError("containment edges use the dedicated parent reference field")


def _geometry_bucket(edge: EdgeRecord) -> int:
    distance = (
        abs(edge.relative_geometry[0]) + abs(edge.relative_geometry[1])
        if edge.relative_geometry
        else 0.0
    )
    return int(round(min(max(distance, 0.0), 1.0) * 15.0))


class SsbCodec:
    """Encode and decode the deterministic CPU SSB grammar."""

    def encode(
        self,
        nodes: Iterable[NodeRecord],
        edges: Iterable[EdgeRecord] = (),
        *,
        selected_node_ids: Iterable[int] | None = None,
        node_order: Iterable[int] | None = None,
    ) -> SsbEncoding:
        """Encode selected nodes and remap all surviving references."""

        node_list = list(nodes)
        node_map = {node.node_id: node for node in node_list}
        if len(node_map) != len(node_list):
            raise ValueError("node IDs must be unique")
        edge_list = list(edges)
        for edge in edge_list:
            if edge.src not in node_map or edge.dst not in node_map:
                raise ValueError(f"edge references a missing node: {edge}")
        selected = set(node_map if selected_node_ids is None else selected_node_ids)
        unknown = selected - set(node_map)
        if unknown:
            raise ValueError(f"selected node IDs are missing from graph: {sorted(unknown)}")
        if node_order is None:
            order = sorted(node_map)
        else:
            requested_order = list(node_order)
            if set(requested_order) != set(node_map):
                raise ValueError("node_order must contain every graph node exactly once")
            if len(requested_order) != len(set(requested_order)):
                raise ValueError("node_order contains duplicates")
            order = requested_order
        ordered_selected = tuple(node_id for node_id in order if node_id in selected)
        node_indices = {node_id: index + 1 for index, node_id in enumerate(ordered_selected)}
        parent_by_child: dict[int, int] = {
            edge.dst: edge.src for edge in edge_list if edge.relation is RelationType.CONTAINMENT
        }
        for child_id, parent_id in parent_by_child.items():
            if child_id in selected and parent_id not in node_map:
                raise ValueError(f"parent reference for node {child_id} is missing")

        edges_by_source: dict[int, list[EdgeRecord]] = {node_id: [] for node_id in ordered_selected}
        for edge in edge_list:
            if edge.src not in selected or edge.relation is RelationType.CONTAINMENT:
                continue
            edges_by_source[edge.src].append(edge)
        for source_id, source_edges in edges_by_source.items():
            if len(source_edges) > MAX_RELATION_SLOTS:
                raise ValueError(
                    f"node {source_id} has {len(source_edges)} non-containment edges; "
                    f"maximum is {MAX_RELATION_SLOTS}"
                )
            source_edges.sort(
                key=lambda edge: (_relation_code(edge), edge.direction.value, edge.dst)
            )

        tokens: list[int] = [MAGIC_TOKEN, CODEC_VERSION, len(ordered_selected)]
        costs: dict[int, int] = {}
        for node_id in ordered_selected:
            node = node_map[node_id]
            node_parent_id: int | None = (
                node.parent_id if node.parent_id is not None else parent_by_child.get(node_id)
            )
            parent_index = (
                node_indices.get(node_parent_id, NULL_REFERENCE)
                if node_parent_id is not None
                else NULL_REFERENCE
            )
            tokens.extend(
                (
                    NODE_START_TOKEN,
                    _NODE_TYPE_CODES[node.node_type],
                    *quantize_box(node.box_xyxy_norm),
                    min(max(node.hierarchy_depth, 0), 1023),
                    _actionability_bits(node),
                    quantize_coordinate(node.detector_confidence),
                    quantize_coordinate(node.ocr_confidence),
                    quantize_coordinate(node.icon_confidence),
                    _icon_class(node),
                    parent_index,
                    TEXT_START_TOKEN,
                    *node.text_token_ids,
                    TEXT_END_TOKEN,
                )
            )
            source_edges = edges_by_source[node_id]
            for edge_index in range(MAX_RELATION_SLOTS):
                if edge_index < len(source_edges):
                    edge = source_edges[edge_index]
                    direction_code = _DIRECTION_CODES.get(edge.direction)
                    if direction_code is None:
                        raise ValueError(f"direction {edge.direction} cannot be serialized")
                    tokens.extend(
                        (
                            RELATION_SLOT_TOKEN,
                            _relation_code(edge),
                            direction_code,
                            node_indices.get(edge.dst, NULL_REFERENCE),
                            _geometry_bucket(edge),
                        )
                    )
                else:
                    tokens.extend((RELATION_SLOT_TOKEN, 0, 0, 0, 0))
            tokens.append(NODE_END_TOKEN)
            costs[node_id] = node_token_cost(node)
        tokens.append(END_TOKEN)
        expected_cost = graph_fixed_token_cost() + sum(costs.values())
        if len(tokens) != expected_cost:
            raise AssertionError(
                f"codec cost mismatch: stream={len(tokens)} expected={expected_cost}"
            )
        return SsbEncoding(
            tokens=tuple(tokens),
            ordered_node_ids=ordered_selected,
            node_costs=costs,
            node_indices=node_indices,
            total_cost=expected_cost,
        )

    def decode(self, tokens: Iterable[int]) -> DecodedSsb:
        """Parse an integer-token stream and reject malformed references."""

        stream = tuple(tokens)
        if len(stream) < HEADER_TOKEN_COUNT + FOOTER_TOKEN_COUNT:
            raise ValueError("SSB stream is shorter than its header and footer")
        if stream[0] != MAGIC_TOKEN:
            raise ValueError("invalid SSB magic token")
        version = stream[1]
        if version != CODEC_VERSION:
            raise ValueError(f"unsupported SSB version {version}")
        node_count = stream[2]
        if node_count < 0:
            raise ValueError("node count cannot be negative")
        position = HEADER_TOKEN_COUNT
        decoded: list[DecodedNode] = []
        for node_index in range(node_count):
            if stream[position] != NODE_START_TOKEN:
                raise ValueError(f"node {node_index} does not start with NODE_START")
            position += 1
            type_code = stream[position]
            position += 1
            try:
                node_type = _NODE_TYPE_FROM_CODE[type_code]
            except KeyError as error:
                raise ValueError(f"unknown node type code {type_code}") from error
            quantized_box = cast(tuple[int, int, int, int], tuple(stream[position : position + 4]))
            if len(quantized_box) != 4 or any(value < 0 or value > 1023 for value in quantized_box):
                raise ValueError("invalid quantized box")
            position += 4
            depth, action_bits, detector_bin, ocr_bin, icon_bin, icon_class, parent_index = stream[
                position : position + 7
            ]
            position += 7
            if not 0 <= parent_index <= node_count:
                raise ValueError("parent reference is outside the graph")
            if stream[position] != TEXT_START_TOKEN:
                raise ValueError("missing TEXT_START token")
            position += 1
            text_ids: list[int] = []
            while stream[position] != TEXT_END_TOKEN:
                if stream[position] < 0:
                    raise ValueError("text token IDs must be non-negative")
                text_ids.append(stream[position])
                position += 1
                if position >= len(stream):
                    raise ValueError("unterminated text token sequence")
            position += 1
            relation_slots: list[DecodedRelation] = []
            for _ in range(MAX_RELATION_SLOTS):
                if stream[position] != RELATION_SLOT_TOKEN:
                    raise ValueError("missing relation slot token")
                relation_code, direction_code, destination_index, geometry_bucket = stream[
                    position + 1 : position + 5
                ]
                position += RELATION_SLOT_TOKENS
                if relation_code == 0:
                    if any((direction_code, destination_index, geometry_bucket)):
                        raise ValueError("null relation slot has non-null fields")
                    relation_slots.append(DecodedRelation(None, None, None, 0))
                    continue
                relation_type = {
                    1: RelationType.PROXIMITY,
                    2: RelationType.ORDINAL,
                    3: RelationType.ORDINAL,
                }.get(relation_code)
                direction = _DIRECTION_FROM_CODE.get(direction_code)
                if relation_type is None or direction is None:
                    raise ValueError("invalid relation slot code")
                if not 0 <= destination_index <= node_count:
                    raise ValueError("relation destination is outside the graph")
                relation_slots.append(
                    DecodedRelation(
                        relation_type,
                        direction,
                        destination_index or None,
                        geometry_bucket,
                    )
                )
            if stream[position] != NODE_END_TOKEN:
                raise ValueError("missing NODE_END token")
            position += 1
            decoded.append(
                DecodedNode(
                    node_index=node_index + 1,
                    node_type=node_type,
                    quantized_box=quantized_box,
                    depth=depth,
                    actionability_bits=action_bits,
                    detector_confidence_bin=detector_bin,
                    ocr_confidence_bin=ocr_bin,
                    icon_confidence_bin=icon_bin,
                    icon_class=icon_class,
                    parent_index=parent_index or None,
                    text_token_ids=tuple(text_ids),
                    relation_slots=tuple(relation_slots),
                )
            )
        if position >= len(stream) or stream[position] != END_TOKEN:
            raise ValueError("missing SSB end token or trailing data")
        position += 1
        for node in decoded:
            if node.parent_index is not None and node.parent_index >= node.node_index:
                raise ValueError("parent reference must point to an earlier serialized node")
            for decoded_relation in node.relation_slots:
                if (
                    decoded_relation.destination_index is not None
                    and decoded_relation.destination_index > node_count
                ):
                    raise ValueError("relation points outside decoded graph")
        return DecodedSsb(version=version, nodes=tuple(decoded), tokens_consumed=position)
