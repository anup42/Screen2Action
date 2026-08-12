"""Exact independent-item budget selection followed by deterministic closure repair."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations

import torch

from screen2action.data.schema import EdgeRecord, NodeRecord, NodeType, RelationType
from screen2action.ssb.codec import graph_fixed_token_cost, node_token_cost


class BudgetError(ValueError):
    """Raised when mandatory graph content cannot fit the SSB budget."""


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Selection mask, exact cost, and closure-repair audit information."""

    selected_node_ids: tuple[int, ...]
    initial_node_ids: tuple[int, ...]
    inserted_closure_ids: tuple[int, ...]
    dropped_reference_ids: tuple[int, ...]
    dropped_selected_ids: tuple[int, ...]
    node_costs: Mapping[int, int]
    total_cost: int
    budget: int


@dataclass(frozen=True, slots=True)
class _DPSolution:
    score: float
    node_ids: tuple[int, ...]


def _default_value(node: NodeRecord) -> float:
    if node.retention_score is not None:
        return node.retention_score
    return max(node.detector_confidence, node.ocr_confidence, node.icon_confidence)


def _better(
    first: _DPSolution,
    second: _DPSolution | None,
    order_rank: Mapping[int, int] | None = None,
) -> bool:
    if second is None:
        return True
    if first.score != second.score:
        return first.score > second.score
    rank = order_rank or {}
    first_key = tuple(rank.get(node_id, node_id) for node_id in first.node_ids)
    second_key = tuple(rank.get(node_id, node_id) for node_id in second.node_ids)
    return first_key < second_key


def _parent_map(nodes: Mapping[int, NodeRecord], edges: Iterable[EdgeRecord]) -> dict[int, int]:
    result = {node.node_id: node.parent_id for node in nodes.values() if node.parent_id is not None}
    for edge in edges:
        if edge.relation is not RelationType.CONTAINMENT:
            continue
        previous = result.get(edge.dst)
        if previous is not None and previous != edge.src:
            raise ValueError(f"node {edge.dst} has conflicting parents")
        result[edge.dst] = edge.src
    for child, parent in result.items():
        if parent not in nodes:
            raise ValueError(f"parent {parent} for node {child} is missing")
    return result


def _ancestors(node_id: int, parent_map: Mapping[int, int]) -> tuple[int, ...]:
    result: list[int] = []
    visited: set[int] = set()
    current = node_id
    while current in parent_map:
        if current in visited:
            raise ValueError("parent map contains a cycle")
        visited.add(current)
        current = parent_map[current]
        result.append(current)
    return tuple(result)


def _normalize_required_ids(
    nodes: Mapping[int, NodeRecord],
    parent_map: Mapping[int, int],
    mandatory_ids: Iterable[int],
    target_ids: Iterable[int],
) -> set[int]:
    required = set(mandatory_ids) | set(target_ids)
    required |= {
        node_id
        for node_id, node in nodes.items()
        if node.mandatory
        or node.node_type is NodeType.ROOT
        or node.is_modal
        or node.is_scroll_container
    }
    unknown = required - set(nodes)
    if unknown:
        raise BudgetError(f"mandatory/target node IDs are missing: {sorted(unknown)}")
    for node_id in tuple(required):
        required.update(_ancestors(node_id, parent_map))
    return required


def _run_dp(
    candidates: list[NodeRecord],
    capacity: int,
    costs: Mapping[int, int],
    values: Mapping[int, float],
    order_rank: Mapping[int, int] | None = None,
) -> tuple[int, ...]:
    if capacity < 0:
        raise BudgetError("negative dynamic-programming capacity")
    dp: list[_DPSolution | None] = [None] * (capacity + 1)
    dp[0] = _DPSolution(0.0, ())
    for node in sorted(candidates, key=lambda item: item.node_id):
        cost = costs[node.node_id]
        value = values[node.node_id]
        for current_capacity in range(capacity, cost - 1, -1):
            previous = dp[current_capacity - cost]
            if previous is None:
                continue
            candidate = _DPSolution(
                previous.score + value,
                previous.node_ids + (node.node_id,),
            )
            if _better(candidate, dp[current_capacity], order_rank):
                dp[current_capacity] = candidate
    best: _DPSolution | None = None
    for solution in dp:
        if solution is not None and _better(solution, best, order_rank):
            best = solution
    return () if best is None else best.node_ids


def brute_force_independent_optimum(
    nodes: Iterable[NodeRecord],
    budget: int,
    *,
    mandatory_ids: Iterable[int] = (),
    values: Mapping[int, float] | None = None,
    costs: Mapping[int, int] | None = None,
) -> tuple[int, ...]:
    """Return the independent knapsack optimum for small test graphs.

    This oracle deliberately excludes closure repair; it tests the exact DP
    portion of BudgetSelect against exhaustive enumeration.
    """

    node_list = list(nodes)
    node_map = {node.node_id: node for node in node_list}
    if len(node_map) != len(node_list):
        raise ValueError("node IDs must be unique")
    cost_map = {node.node_id: node_token_cost(node) for node in node_list}
    if costs is not None:
        cost_map.update(costs)
    value_map = {node.node_id: _default_value(node) for node in node_list}
    if values is not None:
        value_map.update(values)
    mandatory = set(mandatory_ids)
    if not mandatory <= set(node_map):
        raise BudgetError("brute-force mandatory IDs are missing")
    fixed = graph_fixed_token_cost()
    base_cost = fixed + sum(cost_map[node_id] for node_id in mandatory)
    if base_cost > budget:
        raise BudgetError("mandatory nodes exceed budget")
    candidates = [node for node in node_list if node.node_id not in mandatory]
    mandatory_score = sum(value_map[node_id] for node_id in mandatory)
    best_ids: tuple[int, ...] = tuple(sorted(mandatory))
    best_score = mandatory_score
    capacity = budget - base_cost
    for count in range(len(candidates) + 1):
        for chosen in combinations(candidates, count):
            chosen_ids = tuple(sorted(node.node_id for node in chosen))
            if sum(cost_map[node_id] for node_id in chosen_ids) > capacity:
                continue
            score = mandatory_score + sum(value_map[node_id] for node_id in chosen_ids)
            candidate_ids = tuple(sorted(mandatory | set(chosen_ids)))
            if score > best_score or (score == best_score and candidate_ids < best_ids):
                best_score = score
                best_ids = candidate_ids
    return best_ids


def budget_select(
    nodes: Iterable[NodeRecord],
    edges: Iterable[EdgeRecord] = (),
    *,
    budget: int = 512,
    mandatory_ids: Iterable[int] = (),
    target_ids: Iterable[int] = (),
    values: Mapping[int, float] | None = None,
    costs: Mapping[int, int] | None = None,
    node_order: Iterable[int] | None = None,
) -> SelectionResult:
    """Select nodes under budget, then repair parent/reference closure.

    The dynamic program is exact for fixed per-node costs. Closure dependencies
    are then inserted; optional relation-reference nodes are removed from
    lowest to highest value when necessary. The returned mask is deterministic
    for a fixed graph and value mapping.
    """

    if budget <= 0:
        raise BudgetError("budget must be positive")
    node_list = list(nodes)
    if node_order is not None:
        requested_order = tuple(node_order)
        if set(requested_order) != {node.node_id for node in node_list} or len(
            requested_order
        ) != len(node_list):
            raise ValueError("node_order must contain every node exactly once")
        by_id = {node.node_id: node for node in node_list}
        node_list = [by_id[node_id] for node_id in requested_order]
    node_map = {node.node_id: node for node in node_list}
    if len(node_map) != len(node_list):
        raise ValueError("node IDs must be unique")
    edge_list = list(edges)
    parent_map = _parent_map(node_map, edge_list)
    cost_map = {node.node_id: node_token_cost(node) for node in node_list}
    if costs is not None:
        cost_map.update(costs)
    if any(cost_map[node_id] <= 0 for node_id in node_map):
        raise BudgetError("node costs must be positive")
    unknown_costs = set(cost_map) - set(node_map)
    if unknown_costs:
        raise BudgetError(f"costs contain unknown node IDs: {sorted(unknown_costs)}")
    value_map = {node.node_id: _default_value(node) for node in node_list}
    if values is not None:
        value_map.update(values)
    required = _normalize_required_ids(node_map, parent_map, mandatory_ids, target_ids)
    fixed_cost = graph_fixed_token_cost()
    required_cost = fixed_cost + sum(cost_map[node_id] for node_id in required)
    if required_cost > budget:
        raise BudgetError(
            f"mandatory/target closure costs {required_cost} tokens, budget is {budget}"
        )
    capacity = budget - required_cost
    candidates = [node for node in node_list if node.node_id not in required]
    order_rank = {node.node_id: index for index, node in enumerate(node_list)}
    selected_from_dp = set(_run_dp(candidates, capacity, cost_map, value_map, order_rank))
    initial_selected = required | selected_from_dp
    initial_ordered = tuple(
        node_id for node_id in (node.node_id for node in node_list) if node_id in initial_selected
    )

    outgoing_references: dict[int, list[int]] = {node_id: [] for node_id in initial_selected}
    for edge in edge_list:
        if edge.relation is not RelationType.CONTAINMENT and edge.src in initial_selected:
            outgoing_references[edge.src].append(edge.dst)

    closure_ids: set[int] = set()
    for source_id in initial_selected:
        for ancestor_id in _ancestors(source_id, parent_map):
            if ancestor_id not in initial_selected:
                closure_ids.add(ancestor_id)
        for reference_id in outgoing_references.get(source_id, ()):
            if reference_id not in initial_selected:
                closure_ids.add(reference_id)
            for ancestor_id in _ancestors(reference_id, parent_map):
                if ancestor_id not in initial_selected:
                    closure_ids.add(ancestor_id)

    selected = set(initial_selected) | closure_ids
    dropped_reference_ids: set[int] = set()

    def current_cost() -> int:
        return fixed_cost + sum(cost_map[node_id] for node_id in selected)

    optional_closure = sorted(
        closure_ids,
        key=lambda node_id: (value_map[node_id], -node_id),
    )
    for node_id in optional_closure:
        if current_cost() <= budget:
            break
        selected.remove(node_id)
        dropped_reference_ids.add(node_id)
    if current_cost() > budget:
        raise BudgetError("closure repair could not fit the budget without removing required nodes")

    selected_ordered = tuple(
        node_id for node_id in (node.node_id for node in node_list) if node_id in selected
    )
    dropped_selected = tuple(sorted(initial_selected - selected))
    return SelectionResult(
        selected_node_ids=selected_ordered,
        initial_node_ids=initial_ordered,
        inserted_closure_ids=tuple(sorted(closure_ids & selected)),
        dropped_reference_ids=tuple(sorted(dropped_reference_ids)),
        dropped_selected_ids=dropped_selected,
        node_costs=cost_map,
        total_cost=current_cost(),
        budget=budget,
    )


def validate_selection(result: SelectionResult) -> None:
    """Raise if a selection violates its budget or contains duplicate IDs."""

    if len(result.selected_node_ids) != len(set(result.selected_node_ids)):
        raise ValueError("selection contains duplicate node IDs")
    if result.total_cost > result.budget:
        raise BudgetError("selection exceeds its stated budget")
    recomputed = graph_fixed_token_cost() + sum(
        result.node_costs[node_id] for node_id in result.selected_node_ids
    )
    if recomputed != result.total_cost:
        raise ValueError(f"selection cost mismatch: {recomputed} != {result.total_cost}")


def annealed_temperature(
    step: int,
    total_steps: int,
    *,
    start: float = 1.0,
    end: float = 0.1,
    anneal_fraction: float = 0.8,
) -> float:
    """Return the selector temperature after linear first-80%-schedule annealing."""

    if total_steps <= 0 or step < 0 or start <= 0.0 or end <= 0.0:
        raise ValueError("selector schedule arguments are invalid")
    if not 0.0 < anneal_fraction <= 1.0:
        raise ValueError("anneal_fraction must be in (0, 1]")
    progress = min(max(step / max(total_steps * anneal_fraction, 1.0), 0.0), 1.0)
    return start + (end - start) * progress


def straight_through_gumbel_mask(
    retention_logits: torch.Tensor,
    temperature: float,
    *,
    mandatory_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the hard-forward/soft-backward selector mask."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if mandatory_mask is not None and mandatory_mask.shape != retention_logits.shape:
        raise ValueError("mandatory_mask must match retention_logits")
    uniform = torch.rand(
        retention_logits.shape,
        device=retention_logits.device,
        dtype=retention_logits.dtype,
        generator=generator,
    ).clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    soft = torch.sigmoid((retention_logits + gumbel) / temperature)
    hard = (soft >= 0.5).to(soft.dtype)
    if mandatory_mask is not None:
        hard = torch.where(mandatory_mask.bool(), torch.ones_like(hard), hard)
        soft = torch.where(mandatory_mask.bool(), torch.ones_like(soft), soft)
    return hard + soft - soft.detach()
