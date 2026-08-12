from __future__ import annotations

from screen2action.data.schema import RelationAxis, RelationDirection, RelationType
from screen2action.ssb.hierarchy import build_hierarchy
from screen2action.ssb.relations import (
    build_ordinal_edges,
    build_proximity_edges,
)


def test_proximity_is_cardinal_and_bounded(make_node) -> None:
    nodes = [
        make_node(0, (0.40, 0.40, 0.50, 0.50)),
        make_node(1, (0.10, 0.42, 0.20, 0.52)),
        make_node(2, (0.70, 0.42, 0.80, 0.52)),
        make_node(3, (0.42, 0.10, 0.52, 0.20)),
        make_node(4, (0.42, 0.70, 0.52, 0.80)),
        make_node(5, (0.60, 0.60, 0.70, 0.70)),
    ]
    edges = build_proximity_edges(nodes, parent_by_child={})
    center_edges = [edge for edge in edges if edge.src == 0]
    assert len(center_edges) == 4
    assert {edge.direction for edge in center_edges} == {
        RelationDirection.LEFT,
        RelationDirection.RIGHT,
        RelationDirection.ABOVE,
        RelationDirection.BELOW,
    }
    assert all(edge.relation is RelationType.PROXIMITY for edge in edges)


def test_ordinal_rows_and_columns_are_stable(make_node) -> None:
    nodes = [
        make_node(4, (0.55, 0.10, 0.65, 0.20)),
        make_node(2, (0.10, 0.10, 0.20, 0.20)),
        make_node(3, (0.30, 0.10, 0.40, 0.20)),
    ]
    edges = build_ordinal_edges(nodes, parent_by_child={})
    row_edges = [edge for edge in edges if edge.axis is RelationAxis.ROW]
    assert [(edge.src, edge.dst, edge.direction) for edge in row_edges] == [
        (2, 3, RelationDirection.NEXT),
        (3, 2, RelationDirection.PREVIOUS),
        (3, 4, RelationDirection.NEXT),
        (4, 3, RelationDirection.PREVIOUS),
    ]


def test_ancestor_nodes_do_not_get_proximity_edges(make_node) -> None:
    parent = make_node(1, (0.1, 0.1, 0.9, 0.9))
    child = make_node(2, (0.2, 0.2, 0.4, 0.4))
    hierarchy = build_hierarchy([parent, child])
    edges = build_proximity_edges(hierarchy.nodes, parent_by_child=hierarchy.parent_by_child)
    assert all(edge.src not in {hierarchy.root_id, 1} or edge.dst != 2 for edge in edges)
