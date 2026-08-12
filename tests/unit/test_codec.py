from __future__ import annotations

from screen2action.data.schema import EdgeRecord, NodeType, RelationDirection, RelationType
from screen2action.ssb.codec import (
    FIXED_NODE_TOKEN_COUNT,
    SsbCodec,
    graph_fixed_token_cost,
)
from screen2action.ssb.hierarchy import build_hierarchy
from screen2action.ssb.validation import validate_encoding


def test_codec_has_exact_cost_and_round_trips(make_node) -> None:
    hierarchy = build_hierarchy(
        [
            make_node(1, (0.1, 0.1, 0.8, 0.8), node_type=NodeType.CONTAINER),
            make_node(2, (0.2, 0.2, 0.4, 0.4), text_token_ids=(7, 8)),
        ]
    )
    edges = list(hierarchy.containment_edges)
    edges.append(
        EdgeRecord(
            src=1,
            dst=2,
            relation=RelationType.PROXIMITY,
            direction=RelationDirection.RIGHT,
            relative_geometry=(0.1, 0.0),
        )
    )
    codec = SsbCodec()
    encoding = codec.encode(hierarchy.nodes, edges, node_order=hierarchy.depth_first_order)
    assert encoding.total_cost == graph_fixed_token_cost() + sum(encoding.node_costs.values())
    assert encoding.node_costs[2] == FIXED_NODE_TOKEN_COUNT + 2
    decoded = codec.decode(encoding.tokens)
    assert decoded.tokens_consumed == encoding.total_cost
    assert [node.node_type for node in decoded.nodes] == [
        NodeType.ROOT,
        NodeType.CONTAINER,
        NodeType.CONTROL,
    ]
    assert decoded.nodes[2].text_token_ids == (7, 8)
    validate_encoding(encoding, codec)


def test_codec_nulls_omitted_references(make_node) -> None:
    nodes = [make_node(0, (0.0, 0.0, 1.0, 1.0)), make_node(1, (0.1, 0.1, 0.4, 0.4))]
    edge = EdgeRecord(
        src=0,
        dst=1,
        relation=RelationType.PROXIMITY,
        direction=RelationDirection.RIGHT,
    )
    encoding = SsbCodec().encode(nodes, [edge], selected_node_ids=[0])
    decoded = SsbCodec().decode(encoding.tokens)
    assert all(slot.destination_index is None for slot in decoded.nodes[0].relation_slots)
