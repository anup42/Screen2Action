from __future__ import annotations

from screen2action.data.schema import EdgeRecord, RelationDirection, RelationType
from screen2action.ssb.codec import graph_fixed_token_cost
from screen2action.ssb.selector import (
    brute_force_independent_optimum,
    budget_select,
    validate_selection,
)


def test_dynamic_program_matches_brute_force_oracle(make_node) -> None:
    nodes = [
        make_node(
            index, (0.05 * index, 0.0, 0.05 * index + 0.02, 0.02), confidence=0.2 + index * 0.1
        )
        for index in range(4)
    ]
    costs = {node.node_id: cost for node, cost in zip(nodes, (1, 3, 2, 4), strict=True)}
    values = {node.node_id: value for node, value in zip(nodes, (0.2, 0.9, 0.6, 1.2), strict=True)}
    budget = graph_fixed_token_cost() + costs[0] + 5
    result = budget_select(nodes, budget=budget, mandatory_ids=[0], costs=costs, values=values)
    oracle = brute_force_independent_optimum(
        nodes,
        budget,
        mandatory_ids=[0],
        costs=costs,
        values=values,
    )
    assert result.initial_node_ids == oracle
    validate_selection(result)


def test_closure_repair_drops_low_value_reference_when_budget_is_tight(make_node) -> None:
    root = make_node(0, (0.0, 0.0, 1.0, 1.0), mandatory=True)
    target = make_node(1, (0.1, 0.1, 0.4, 0.4), confidence=0.9)
    reference = make_node(2, (0.5, 0.1, 0.8, 0.4), confidence=0.1)
    containment = EdgeRecord(0, 1, RelationType.CONTAINMENT, RelationDirection.CONTAINS)
    proximity = EdgeRecord(1, 2, RelationType.PROXIMITY, RelationDirection.RIGHT)
    costs = {0: 1, 1: 2, 2: 2}
    budget = graph_fixed_token_cost() + costs[0] + costs[1]
    result = budget_select(
        [root, target, reference],
        [containment, proximity],
        budget=budget,
        target_ids=[1],
        costs=costs,
    )
    assert result.selected_node_ids == (0, 1)
    assert result.inserted_closure_ids == ()
    assert result.dropped_reference_ids == (2,)
    validate_selection(result)
