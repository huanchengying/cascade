#!/usr/bin/env python3
"""Shared-prefix batch optimization for weighted temporal spanning trees.

Independent shortest paths can miss a useful but individually expensive early
prefix.  This module evaluates a root-to-``(hub, arrival)`` prefix once and
reroutes a batch of terminals through that state.  The physical prefix edges
are charged once in the union, so a move can improve the tree even when no
single terminal would choose the prefix by itself.

The method is a monotone primal heuristic, not an exact algorithm and not yet a
novelty claim.  Every accepted candidate is converted to a physical temporal
tree and validated against the original graph.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from temporal_mst_prototype import (
    TemporalEdge,
    TemporalGraph,
    _tree_from_edge_ids,
    earliest_arrival_tree,
    temporal_reachable,
    validate_temporal_tree,
)


@dataclass(frozen=True)
class _Label:
    vertex: str
    arrival: float
    cost: float
    parent: Optional[int]
    edge_id: str


@dataclass
class _ShortestPathIndex:
    source: str
    source_time: float
    labels: List[_Label]
    labels_by_vertex: Dict[str, List[int]]
    best_label: Dict[str, int]
    best_cost: Dict[str, float]

    def path(self, vertex: str, label_id: Optional[int] = None) -> List[str]:
        if vertex == self.source and label_id is None:
            return []
        if label_id is None:
            label_id = self.best_label.get(vertex)
        if label_id is None:
            return []
        path: List[str] = []
        while label_id is not None:
            label = self.labels[label_id]
            path.append(label.edge_id)
            label_id = label.parent
        path.reverse()
        return path


@dataclass(frozen=True)
class _HubState:
    vertex: str
    arrival: float
    prefix_cost: float
    label_id: int
    fanout: int


@dataclass
class SharedPrefixResult:
    cost: float
    edge_ids: List[str]
    reachable: List[str]
    baseline_cost: float
    baseline_edge_ids: List[str]
    exchange_only_cost: float
    shared_only_cost: float
    relative_improvement: float
    rounds: int
    accepted_moves: int
    accepted_edge_swaps: int
    evaluated_hub_states: int
    final_strategy: str
    algorithm: str = "shared-prefix-batch"


def _edge_cost_map(graph: TemporalGraph) -> Dict[str, float]:
    return {edge.eid: edge.weight for edge in graph.edges}


def _shortest_temporal_paths(
    graph: TemporalGraph,
    source: str,
    *,
    source_time: float = -math.inf,
    edge_costs: Optional[Dict[str, float]] = None,
) -> _ShortestPathIndex:
    """Compute cheapest temporal paths with a chronological label sweep.

    Strictly positive durations make the event order acyclic.  For each tail
    vertex, ``available_cost`` is the cheapest label that has arrived by the
    current departure time; more expensive early labels remain implicitly in
    the prefix minimum until a cheaper label becomes available.
    """

    if source not in graph.vertices:
        raise ValueError(f"unknown source vertex: {source}")
    if any(edge.arrival <= edge.start for edge in graph.edges):
        raise ValueError("shared-prefix optimization requires positive durations")
    costs = edge_costs or _edge_cost_map(graph)
    if any(not math.isfinite(costs.get(edge.eid, edge.weight)) for edge in graph.edges):
        raise ValueError("temporal shortest-path costs must be finite")

    ordered = sorted(
        (edge for edge in graph.edges if edge.start >= source_time and edge.v != source),
        key=lambda edge: (edge.start, edge.arrival, edge.eid),
    )
    labels: List[_Label] = []
    labels_by_vertex: Dict[str, List[int]] = defaultdict(list)
    best_label: Dict[str, int] = {}
    best_terminal_cost: Dict[str, float] = {source: 0.0}
    available_cost: Dict[str, float] = {source: 0.0}
    available_label: Dict[str, Optional[int]] = {source: None}
    pending_best: Dict[Tuple[str, float], int] = {}
    pending: List[Tuple[float, int, str]] = []

    for edge in ordered:
        while pending and pending[0][0] <= edge.start:
            arrival, label_id, vertex = heapq.heappop(pending)
            if pending_best.get((vertex, arrival)) != label_id:
                continue
            del pending_best[vertex, arrival]
            label = labels[label_id]
            old_cost = available_cost.get(vertex, math.inf)
            if label.cost < old_cost:
                available_cost[vertex] = label.cost
                available_label[vertex] = label_id

        prefix_cost = available_cost.get(edge.u, math.inf)
        if prefix_cost == math.inf:
            continue
        candidate_cost = prefix_cost + costs.get(edge.eid, edge.weight)
        key = (edge.v, edge.arrival)
        previous_id = pending_best.get(key)
        if previous_id is not None and labels[previous_id].cost <= candidate_cost:
            continue
        label_id = len(labels)
        labels.append(
            _Label(
                vertex=edge.v,
                arrival=edge.arrival,
                cost=candidate_cost,
                parent=available_label.get(edge.u),
                edge_id=edge.eid,
            )
        )
        labels_by_vertex[edge.v].append(label_id)
        pending_best[key] = label_id
        heapq.heappush(pending, (edge.arrival, label_id, edge.v))
        previous_best = best_terminal_cost.get(edge.v, math.inf)
        previous_label = best_label.get(edge.v)
        if (
            candidate_cost < previous_best
            or (
                math.isclose(candidate_cost, previous_best)
                and previous_label is not None
                and edge.arrival < labels[previous_label].arrival
            )
        ):
            best_terminal_cost[edge.v] = candidate_cost
            best_label[edge.v] = label_id

    return _ShortestPathIndex(
        source=source,
        source_time=source_time,
        labels=labels,
        labels_by_vertex=dict(labels_by_vertex),
        best_label=best_label,
        best_cost=best_terminal_cost,
    )


def _tree_paths(
    root: str,
    tree: Sequence[TemporalEdge],
    required: Iterable[str],
) -> Dict[str, List[str]]:
    parent = {edge.v: edge for edge in tree}
    memo: Dict[str, List[str]] = {root: []}

    def path(vertex: str, active: Set[str]) -> List[str]:
        if vertex in memo:
            return memo[vertex]
        if vertex in active or vertex not in parent:
            raise RuntimeError("selected edges do not form a rooted tree")
        edge = parent[vertex]
        prefix = path(edge.u, active | {vertex})
        memo[vertex] = prefix + [edge.eid]
        return memo[vertex]

    for vertex in required:
        path(vertex, set())
    return memo


def _pareto_hub_states(
    graph: TemporalGraph,
    index: _ShortestPathIndex,
    *,
    max_states_per_vertex: int,
    max_hub_states: int,
) -> List[_HubState]:
    outgoing: Dict[str, List[TemporalEdge]] = defaultdict(list)
    for edge in graph.edges:
        outgoing[edge.u].append(edge)
    candidates: List[_HubState] = []
    for vertex, label_ids in index.labels_by_vertex.items():
        frontier: List[int] = []
        best_cost = math.inf
        for label_id in sorted(
            label_ids,
            key=lambda item: (
                index.labels[item].arrival,
                index.labels[item].cost,
                item,
            ),
        ):
            label = index.labels[label_id]
            if label.cost + 1e-12 < best_cost:
                frontier.append(label_id)
                best_cost = label.cost
        if len(frontier) > max_states_per_vertex:
            if max_states_per_vertex == 1:
                frontier = [frontier[0]]
            else:
                positions = {
                    round(position * (len(frontier) - 1) / (max_states_per_vertex - 1))
                    for position in range(max_states_per_vertex)
                }
                frontier = [frontier[position] for position in sorted(positions)]
        for label_id in frontier:
            label = index.labels[label_id]
            feasible_edges = [
                edge for edge in outgoing.get(vertex, [])
                if edge.start >= label.arrival
            ]
            fanout = len({edge.v for edge in feasible_edges if edge.v != vertex})
            if fanout >= 2:
                candidates.append(
                    _HubState(vertex, label.arrival, label.cost, label_id, fanout)
                )
    candidates.sort(
        key=lambda state: (-state.fanout, state.arrival, state.prefix_cost, state.vertex)
    )
    return candidates[:max_hub_states]


def _union_cost(paths: Iterable[Sequence[str]], weights: Dict[str, float]) -> float:
    return math.fsum(weights[edge_id] for edge_id in {eid for path in paths for eid in path})


def _parent_exchange_descent(
    graph: TemporalGraph,
    root: str,
    required: Set[str],
    tree: Sequence[TemporalEdge],
) -> Tuple[List[TemporalEdge], int]:
    """Apply improving one-edge reparenting moves that preserve all deadlines."""

    parent_edge = {edge.v: edge for edge in tree}
    children: Dict[str, Dict[str, TemporalEdge]] = defaultdict(dict)
    for edge in tree:
        children[edge.u][edge.eid] = edge
    incoming: Dict[str, List[TemporalEdge]] = defaultdict(list)
    for edge in graph.edges:
        if edge.u in required and edge.v in required and edge.v != root:
            incoming[edge.v].append(edge)
    for edges in incoming.values():
        edges.sort(key=lambda edge: (edge.weight, edge.arrival, edge.eid))
    arrival = {root: -math.inf}
    arrival.update({vertex: edge.arrival for vertex, edge in parent_edge.items()})
    accepted = 0
    while True:
        changed = False

        def is_descendant(candidate: str, ancestor: str) -> bool:
            cursor = candidate
            while cursor != root and cursor in parent_edge:
                if cursor == ancestor:
                    return True
                cursor = parent_edge[cursor].u
            return cursor == ancestor

        for vertex in sorted(required - {root}):
            old = parent_edge[vertex]
            deadline = min(
                (child.start for child in children.get(vertex, {}).values()),
                default=math.inf,
            )
            replacement = None
            for edge in incoming.get(vertex, []):
                if edge.weight >= old.weight:
                    break
                if edge.eid == old.eid:
                    continue
                if arrival.get(edge.u, math.inf) > edge.start:
                    continue
                if edge.arrival > deadline or is_descendant(edge.u, vertex):
                    continue
                replacement = edge
                break
            if replacement is None:
                continue
            del children[old.u][old.eid]
            children[replacement.u][replacement.eid] = replacement
            parent_edge[vertex] = replacement
            arrival[vertex] = replacement.arrival
            old = replacement
            accepted += 1
            changed = True
        if not changed:
            break
    current = list(parent_edge.values())
    if not validate_temporal_tree(graph, root, current, required):
        raise RuntimeError("deadline-safe parent exchange failed validation")
    return current, accepted


def _best_hub_move(
    graph: TemporalGraph,
    root: str,
    required: Set[str],
    current_tree: Sequence[TemporalEdge],
    current_paths: Dict[str, List[str]],
    root_index: _ShortestPathIndex,
    hub: _HubState,
    *,
    evaluated_prefixes: int,
) -> Tuple[float, List[TemporalEdge]]:
    weights = _edge_cost_map(graph)
    current_edge_ids = {edge.eid for edge in current_tree}
    marginal_costs = {
        edge.eid: (0.0 if edge.eid in current_edge_ids else edge.weight)
        for edge in graph.edges
    }
    suffix_index = _shortest_temporal_paths(
        graph,
        hub.vertex,
        source_time=hub.arrival,
        edge_costs=marginal_costs,
    )
    prefix = root_index.path(hub.vertex, hub.label_id)
    alternatives: Dict[str, List[str]] = {}
    scores: Dict[str, float] = {}
    for terminal in required - {root}:
        if terminal == hub.vertex:
            suffix: List[str] = []
            suffix_cost = 0.0
        elif terminal in suffix_index.best_label:
            suffix = suffix_index.path(terminal)
            suffix_cost = suffix_index.best_cost[terminal]
        else:
            continue
        alternative = prefix + suffix
        if alternative == current_paths[terminal]:
            continue
        alternatives[terminal] = alternative
        current_path_cost = math.fsum(weights[edge_id] for edge_id in current_paths[terminal])
        scores[terminal] = current_path_cost - suffix_cost
    if not alternatives:
        return math.inf, []

    order = sorted(
        alternatives,
        key=lambda terminal: (-scores[terminal], len(alternatives[terminal]), terminal),
    )
    counts = Counter(edge_id for path in current_paths.values() for edge_id in path)
    running_cost = math.fsum(weights[edge_id] for edge_id in counts)
    snapshots: List[Tuple[float, int]] = []
    for position, terminal in enumerate(order, start=1):
        old_edges = set(current_paths[terminal])
        new_edges = set(alternatives[terminal])
        for edge_id in old_edges - new_edges:
            counts[edge_id] -= 1
            if counts[edge_id] == 0:
                running_cost -= weights[edge_id]
        for edge_id in new_edges - old_edges:
            if counts[edge_id] == 0:
                running_cost += weights[edge_id]
            counts[edge_id] += 1
        snapshots.append((running_cost, position))

    candidate_positions = {
        position
        for _, position in sorted(snapshots)[: max(1, evaluated_prefixes)]
    }
    candidate_positions.add(len(order))
    best_cost = math.inf
    best_tree: List[TemporalEdge] = []
    for position in sorted(candidate_positions):
        switched = set(order[:position])
        candidate_paths = [
            alternatives.get(terminal, current_paths[terminal])
            if terminal in switched
            else current_paths[terminal]
            for terminal in required - {root}
        ]
        edge_ids = {edge_id for path in candidate_paths for edge_id in path}
        tree = _tree_from_edge_ids(graph, root, edge_ids, required)
        if not tree:
            continue
        cost = math.fsum(edge.weight for edge in tree)
        if cost < best_cost:
            best_cost = cost
            best_tree = tree
    return best_cost, best_tree


def shortest_path_union_tree(
    graph: TemporalGraph,
    root: str,
) -> Tuple[float, List[TemporalEdge], _ShortestPathIndex]:
    """Construct the independent-shortest-path union baseline."""

    required = temporal_reachable(graph, root, graph.edges)
    index = _shortest_temporal_paths(graph, root)
    paths = []
    for terminal in required - {root}:
        if terminal not in index.best_label:
            raise RuntimeError(f"reachable terminal {terminal} has no weighted path label")
        paths.append(index.path(terminal))
    edge_ids = {edge_id for path in paths for edge_id in path}
    tree = _tree_from_edge_ids(graph, root, edge_ids, required)
    if not validate_temporal_tree(graph, root, tree, required):
        raise RuntimeError("shortest-path union did not reconstruct a temporal tree")
    return math.fsum(edge.weight for edge in tree), tree, index


def shared_prefix_optimize(
    graph: TemporalGraph,
    root: str,
    *,
    max_rounds: int = 3,
    max_hub_states: int = 64,
    max_states_per_vertex: int = 4,
    evaluated_prefixes: int = 8,
) -> SharedPrefixResult:
    """Improve a path-union tree with shared-prefix batch rerouting."""

    if max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    if min(max_hub_states, max_states_per_vertex, evaluated_prefixes) < 1:
        raise ValueError("hub and prefix limits must be positive")
    required = temporal_reachable(graph, root, graph.edges)
    baseline_cost, baseline_tree, _ = shortest_path_union_tree(graph, root)
    exchange_only_tree, exchange_only_swaps = _parent_exchange_descent(
        graph, root, required, baseline_tree
    )
    exchange_only_cost = math.fsum(edge.weight for edge in exchange_only_tree)
    current_tree = list(baseline_tree)
    current_cost = baseline_cost
    evaluated = 0
    accepted = 0
    accepted_swaps = 0
    completed_rounds = 0

    for round_number in range(1, max_rounds + 1):
        completed_rounds = round_number
        root_index = _shortest_temporal_paths(graph, root)
        hubs = _pareto_hub_states(
            graph,
            root_index,
            max_states_per_vertex=max_states_per_vertex,
            max_hub_states=max_hub_states,
        )
        current_paths = _tree_paths(root, current_tree, required)
        best_cost = current_cost
        best_tree: List[TemporalEdge] = []
        for hub in hubs:
            evaluated += 1
            candidate_cost, candidate_tree = _best_hub_move(
                graph,
                root,
                required,
                current_tree,
                current_paths,
                root_index,
                hub,
                evaluated_prefixes=evaluated_prefixes,
            )
            if candidate_tree and candidate_cost + 1e-9 < best_cost:
                best_cost = candidate_cost
                best_tree = candidate_tree
        if not best_tree:
            break
        current_tree = best_tree
        current_cost = best_cost
        accepted += 1

    shared_only_cost = current_cost
    combined_tree, combined_swaps = _parent_exchange_descent(
        graph, root, required, current_tree
    )
    combined_cost = math.fsum(edge.weight for edge in combined_tree)
    if exchange_only_cost + 1e-9 < combined_cost:
        current_tree = exchange_only_tree
        current_cost = exchange_only_cost
        accepted_swaps = exchange_only_swaps
        final_strategy = "exchange-only"
    else:
        current_tree = combined_tree
        current_cost = combined_cost
        accepted_swaps = combined_swaps
        final_strategy = "shared-then-exchange"

    if not validate_temporal_tree(graph, root, current_tree, required):
        raise RuntimeError("shared-prefix optimizer returned an invalid tree")
    improvement = (
        (baseline_cost - current_cost) / baseline_cost if baseline_cost > 0 else 0.0
    )
    return SharedPrefixResult(
        cost=current_cost,
        edge_ids=sorted(edge.eid for edge in current_tree),
        reachable=sorted(required),
        baseline_cost=baseline_cost,
        baseline_edge_ids=sorted(edge.eid for edge in baseline_tree),
        exchange_only_cost=exchange_only_cost,
        shared_only_cost=shared_only_cost,
        relative_improvement=improvement,
        rounds=completed_rounds,
        accepted_moves=accepted,
        accepted_edge_swaps=accepted_swaps,
        evaluated_hub_states=evaluated,
        final_strategy=final_strategy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-hub-states", type=int, default=64)
    parser.add_argument("--max-states-per-vertex", type=int, default=4)
    parser.add_argument("--evaluated-prefixes", type=int, default=8)
    args = parser.parse_args()
    graph = TemporalGraph.from_json(args.input)
    result = shared_prefix_optimize(
        graph,
        args.root,
        max_rounds=args.max_rounds,
        max_hub_states=args.max_hub_states,
        max_states_per_vertex=args.max_states_per_vertex,
        evaluated_prefixes=args.evaluated_prefixes,
    )
    print(json.dumps(asdict(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
