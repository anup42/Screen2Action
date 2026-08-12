from __future__ import annotations

from screen2action.data.schema import NodeType
from screen2action.ssb.hierarchy import build_hierarchy


def test_smallest_qualifying_parent_root_and_depth_first_order(make_node) -> None:
    card = make_node(10, (0.1, 0.1, 0.9, 0.9), node_type=NodeType.CONTAINER)
    nested = make_node(11, (0.2, 0.2, 0.8, 0.8), node_type=NodeType.CONTAINER)
    button = make_node(12, (0.3, 0.3, 0.5, 0.5), node_type=NodeType.CONTROL)
    hierarchy = build_hierarchy([button, card, nested])

    assert hierarchy.root_id == 13
    assert hierarchy.parent_by_child[10] == hierarchy.root_id
    assert hierarchy.parent_by_child[11] == 10
    assert hierarchy.parent_by_child[12] == 11
    assert hierarchy.depth_by_node[12] == 3
    assert hierarchy.depth_first_order == (13, 10, 11, 12)
    root = next(node for node in hierarchy.nodes if node.node_id == hierarchy.root_id)
    assert root.mandatory and root.node_type is NodeType.ROOT


def test_root_is_not_duplicated(make_node) -> None:
    root = make_node(0, (0.0, 0.0, 1.0, 1.0), node_type=NodeType.ROOT)
    hierarchy = build_hierarchy([root])
    assert hierarchy.root_id == 0
    assert len(hierarchy.nodes) == 1
    assert hierarchy.nodes[0].mandatory
