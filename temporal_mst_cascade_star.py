#!/usr/bin/env python3
"""Shared-activation star search with exact batch-relaxed deadlines.

For a fixed incumbent tree, hub, and replacement parent of that hub, this
module optimizes the complete single-hub star neighborhood.  Every other
vertex may retain its incumbent parent or use an event directly from the hub.
Unlike the conservative star search, removing an incumbent child event also
removes that event from its old parent's departure deadline.  A bottom-up tree
DP accounts for these dependencies exactly.

Exhaustive parent evaluation is polynomial but expensive.  The public search
therefore supports explicit hub and parent-candidate caps.  These caps affect
which hub parents are inspected, not the exactness of the DP for an inspected
parent.  Uncapped mode is used by differential tests.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from temporal_mst_prototype import TemporalEdge, TemporalGraph, temporal_reachable, validate_temporal_tree
from temporal_mst_shared import _parent_exchange_descent, shortest_path_union_tree
from temporal_mst_star import TOLERANCE, _TreeIndex, _candidate_groups, _shared_activation_descent


@dataclass
class CascadeStarResult:
    cost: float
    edge_ids: List[str]
    reachable: List[str]
    baseline_cost: float
    exchange_cost: float
    conservative_star_cost: float
    relative_improvement: float
    rounds: int
    accepted_moves: int
    activated_children: int
    relaxed_deadline_vertices: int
    candidate_hubs: int
    candidate_parent_events: int
    evaluated_parent_events: int
    exhaustive_parent_search: bool
    algorithm: str = "batch-relaxed-cascade-star"


@dataclass(frozen=True)
class _ScoredParent:
    upper_saving: float
    hub: str
    edge: TemporalEdge


@dataclass(frozen=True)
class _CascadeMove:
    saving: float
    hub: str
    parent_edge: TemporalEdge
    edge_ids: Tuple[str, ...]
    activated_children: int
    relaxed_deadlines: int


def _scored_parents(
    graph: TemporalGraph,
    root: str,
    required: Set[str],
    tree: Sequence[TemporalEdge],
) -> Tuple[List[_ScoredParent], Dict[str, List[TemporalEdge]]]:
    index = _TreeIndex(root, tree, required)
    incoming, outgoing = _candidate_groups(graph, required)
    scored: List[_ScoredParent] = []
    for hub in sorted(required - {root}):
        parents = [
            edge
            for edge in incoming.get(hub, [])
            if index.arrival[edge.u] <= edge.start
            and not index.is_ancestor(hub, edge.u)
        ]
        if not parents:
            continue
        children = [
            edge
            for edge in outgoing.get(hub, [])
            if edge.v not in {root, hub}
        ]
        best_weight: Dict[str, float] = {}
        total_gain = 0.0
        child_offset = 0
        for parent_edge in parents:
            while (
                child_offset < len(children)
                and children[child_offset].start >= parent_edge.arrival
            ):
                edge = children[child_offset]
                child_offset += 1
                old_weight = best_weight.get(edge.v, math.inf)
                if edge.weight >= old_weight:
                    continue
                old_gain = max(
                    0.0,
                    index.parent_edge[edge.v].weight - old_weight,
                )
                new_gain = max(
                    0.0,
                    index.parent_edge[edge.v].weight - edge.weight,
                )
                best_weight[edge.v] = edge.weight
                total_gain += new_gain - old_gain
            upper_saving = (
                index.parent_edge[hub].weight
                - parent_edge.weight
                + total_gain
            )
            if upper_saving > TOLERANCE:
                scored.append(_ScoredParent(upper_saving, hub, parent_edge))
    scored.sort(key=lambda item: (-item.upper_saving, item.hub, item.edge.eid))
    return scored, outgoing


def _candidate_subset(
    candidates: Sequence[_ScoredParent],
    *,
    max_hubs: Optional[int],
    max_parent_evaluations: Optional[int],
) -> List[_ScoredParent]:
    if max_hubs is not None:
        hub_score: Dict[str, float] = {}
        for candidate in candidates:
            hub_score[candidate.hub] = max(
                hub_score.get(candidate.hub, -math.inf),
                candidate.upper_saving,
            )
        hubs = {
            hub
            for hub, _ in sorted(
                hub_score.items(), key=lambda item: (-item[1], item[0])
            )[:max_hubs]
        }
        candidates = [candidate for candidate in candidates if candidate.hub in hubs]
    if max_parent_evaluations is not None:
        candidates = candidates[:max_parent_evaluations]
    return list(candidates)


def _threshold_cost(
    base: float,
    starts: Sequence[float],
    prefix_delta: Sequence[float],
    prefix_blocked: Sequence[int],
    threshold: float,
) -> float:
    count = bisect.bisect_left(starts, threshold)
    if prefix_blocked[count]:
        return math.inf
    return base + prefix_delta[count]


def _evaluate_parent(
    root: str,
    required: Set[str],
    tree: Sequence[TemporalEdge],
    index: _TreeIndex,
    outgoing: Dict[str, List[TemporalEdge]],
    hub: str,
    parent_edge: TemporalEdge,
) -> Optional[_CascadeMove]:
    forbidden: Set[str] = set()
    cursor = parent_edge.u
    while True:
        forbidden.add(cursor)
        if cursor == root:
            break
        cursor = index.parent[cursor]

    active_by_target: Dict[str, List[TemporalEdge]] = {}
    for edge in outgoing.get(hub, []):
        if edge.start < parent_edge.arrival:
            break
        if edge.v in {root, hub} or edge.v in forbidden:
            continue
        active_by_target.setdefault(edge.v, []).append(edge)

    keep: Dict[str, float] = {}
    replace: Dict[str, float] = {}
    optional: Dict[str, float] = {}
    use_replace: Dict[str, bool] = {}
    best_edge: Dict[str, TemporalEdge] = {}
    starts_by_vertex: Dict[str, List[float]] = {}
    prefix_delta_by_vertex: Dict[str, List[float]] = {}
    prefix_blocked_by_vertex: Dict[str, List[int]] = {}
    base_by_vertex: Dict[str, float] = {}

    for vertex in index.postorder:
        children = index.children[vertex]
        base = math.fsum(optional[child] for child in children)
        entries = [
            (
                index.parent_edge[child].start,
                replace[child],
                optional[child],
            )
            for child in index.children_by_start[vertex]
        ]
        starts: List[float] = []
        prefix_delta = [0.0]
        prefix_blocked = [0]
        for start, forced_cost, optional_cost in entries:
            starts.append(start)
            blocked = prefix_blocked[-1] + int(not math.isfinite(forced_cost))
            delta = 0.0 if not math.isfinite(forced_cost) else forced_cost - optional_cost
            prefix_delta.append(prefix_delta[-1] + delta)
            prefix_blocked.append(blocked)
        starts_by_vertex[vertex] = starts
        prefix_delta_by_vertex[vertex] = prefix_delta
        prefix_blocked_by_vertex[vertex] = prefix_blocked
        base_by_vertex[vertex] = base

        if vertex == hub:
            child_cost = _threshold_cost(
                base,
                starts,
                prefix_delta,
                prefix_blocked,
                parent_edge.arrival,
            )
            hub_cost = (
                math.inf
                if not math.isfinite(child_cost)
                else parent_edge.weight - index.parent_edge[hub].weight + child_cost
            )
            keep[vertex] = hub_cost
            replace[vertex] = hub_cost
            optional[vertex] = hub_cost
            use_replace[vertex] = True
            continue

        keep[vertex] = base
        replace_cost = math.inf
        replacement: Optional[TemporalEdge] = None
        if vertex not in forbidden and vertex != root:
            for edge in active_by_target.get(vertex, []):
                child_cost = _threshold_cost(
                    base,
                    starts,
                    prefix_delta,
                    prefix_blocked,
                    edge.arrival,
                )
                if not math.isfinite(child_cost):
                    continue
                candidate_cost = (
                    edge.weight - index.parent_edge[vertex].weight + child_cost
                )
                if replacement is None or (
                    candidate_cost,
                    edge.weight,
                    edge.arrival,
                    edge.eid,
                ) < (
                    replace_cost,
                    replacement.weight,
                    replacement.arrival,
                    replacement.eid,
                ):
                    replace_cost = candidate_cost
                    replacement = edge
        replace[vertex] = replace_cost
        if replacement is not None:
            best_edge[vertex] = replacement
        if replace_cost + TOLERANCE < base:
            optional[vertex] = replace_cost
            use_replace[vertex] = True
        else:
            optional[vertex] = base
            use_replace[vertex] = False

    if not math.isfinite(optional[root]) or optional[root] >= -TOLERANCE:
        return None

    parent = dict(index.parent_edge)
    stack: List[Tuple[str, str]] = [(root, "keep")]
    while stack:
        vertex, mode = stack.pop()
        if vertex == hub:
            parent[hub] = parent_edge
            threshold: Optional[float] = parent_edge.arrival
        elif mode == "replace":
            selected = best_edge[vertex]
            parent[vertex] = selected
            threshold = selected.arrival
        else:
            threshold = None
        for child in reversed(index.children[vertex]):
            if child == hub:
                child_mode = "hub"
            elif (
                threshold is not None
                and index.parent_edge[child].start < threshold
            ):
                child_mode = "replace"
            else:
                child_mode = "replace" if use_replace[child] else "keep"
            stack.append((child, child_mode))

    candidate = list(parent.values())
    old_cost = math.fsum(edge.weight for edge in tree)
    new_cost = math.fsum(edge.weight for edge in candidate)
    saving = old_cost - new_cost
    if saving <= TOLERANCE:
        return None
    activated = sum(
        edge.eid != index.parent_edge[vertex].eid
        for vertex, edge in parent.items()
        if vertex != hub
    )
    relaxed = int(parent_edge.arrival > index.deadline[hub] + TOLERANCE)
    relaxed += sum(
        edge.eid != index.parent_edge[vertex].eid
        and edge.arrival > index.deadline[vertex] + TOLERANCE
        for vertex, edge in parent.items()
        if vertex != hub
    )
    return _CascadeMove(
        saving,
        hub,
        parent_edge,
        tuple(sorted(edge.eid for edge in candidate)),
        activated,
        relaxed,
    )


def _best_cascade_move(
    graph: TemporalGraph,
    root: str,
    required: Set[str],
    tree: Sequence[TemporalEdge],
    *,
    max_hubs: Optional[int] = None,
    max_parent_evaluations: Optional[int] = None,
) -> Tuple[Optional[_CascadeMove], int, int, int]:
    if max_hubs is not None and max_hubs < 1:
        raise ValueError("max_hubs must be positive or None")
    if max_parent_evaluations is not None and max_parent_evaluations < 1:
        raise ValueError("max_parent_evaluations must be positive or None")
    index = _TreeIndex(root, tree, required)
    candidates, outgoing = _scored_parents(graph, root, required, tree)
    selected = _candidate_subset(
        candidates,
        max_hubs=max_hubs,
        max_parent_evaluations=max_parent_evaluations,
    )
    best: Optional[_CascadeMove] = None
    for candidate in selected:
        move = _evaluate_parent(
            root,
            required,
            tree,
            index,
            outgoing,
            candidate.hub,
            candidate.edge,
        )
        if move is None:
            continue
        if best is None or (
            -move.saving,
            move.hub,
            move.parent_edge.eid,
            move.edge_ids,
        ) < (
            -best.saving,
            best.hub,
            best.parent_edge.eid,
            best.edge_ids,
        ):
            best = move
    return best, len({candidate.hub for candidate in candidates}), len(candidates), len(selected)


def _apply_cascade_move(
    graph: TemporalGraph,
    root: str,
    required: Set[str],
    tree: Sequence[TemporalEdge],
    move: _CascadeMove,
) -> Tuple[float, List[TemporalEdge]]:
    by_id = {edge.eid: edge for edge in graph.edges}
    candidate = [by_id[edge_id] for edge_id in move.edge_ids]
    if not validate_temporal_tree(graph, root, candidate, required):
        raise RuntimeError("cascade-star move failed physical-tree validation")
    old_cost = math.fsum(edge.weight for edge in tree)
    cost = math.fsum(edge.weight for edge in candidate)
    if not math.isclose(old_cost - cost, move.saving, rel_tol=1e-8, abs_tol=1e-7):
        raise RuntimeError("cascade-star DP and reconstructed move disagree")
    return cost, candidate


def cascade_activation_star(
    graph: TemporalGraph,
    root: str,
    *,
    conservative_rounds: int = 4,
    cascade_rounds: int = 2,
    max_hubs: Optional[int] = 64,
    max_parent_evaluations: Optional[int] = 128,
) -> CascadeStarResult:
    if conservative_rounds < 0 or cascade_rounds < 0:
        raise ValueError("round counts must be nonnegative")
    if any(edge.arrival <= edge.start for edge in graph.edges):
        raise ValueError("cascade star requires positive durations")
    required = temporal_reachable(graph, root, graph.edges)
    baseline_cost, baseline, _ = shortest_path_union_tree(graph, root)
    exchanged, _ = _parent_exchange_descent(
        graph, root, required, baseline
    )
    exchange_cost = math.fsum(edge.weight for edge in exchanged)
    if conservative_rounds:
        conservative = _shared_activation_descent(
            graph,
            root,
            required,
            exchanged,
            max_rounds=conservative_rounds,
        )
        tree = conservative.tree
        conservative_cost = conservative.cost
    else:
        tree = exchanged
        conservative_cost = exchange_cost
    cost = conservative_cost
    accepted = 0
    activated = 0
    relaxed = 0
    candidate_hubs = 0
    candidate_parents = 0
    evaluated_parents = 0
    rounds = 0
    for rounds in range(1, cascade_rounds + 1):
        move, hubs, parents, evaluated = _best_cascade_move(
            graph,
            root,
            required,
            tree,
            max_hubs=max_hubs,
            max_parent_evaluations=max_parent_evaluations,
        )
        candidate_hubs += hubs
        candidate_parents += parents
        evaluated_parents += evaluated
        if move is None:
            break
        cost, tree = _apply_cascade_move(graph, root, required, tree, move)
        accepted += 1
        activated += move.activated_children
        relaxed += move.relaxed_deadlines
    if not validate_temporal_tree(graph, root, tree, required):
        raise RuntimeError("cascade-star result is not a temporal tree")
    return CascadeStarResult(
        cost,
        sorted(edge.eid for edge in tree),
        sorted(required),
        baseline_cost,
        exchange_cost,
        conservative_cost,
        (baseline_cost - cost) / baseline_cost if baseline_cost else 0.0,
        rounds,
        accepted,
        activated,
        relaxed,
        candidate_hubs,
        candidate_parents,
        evaluated_parents,
        max_hubs is None and max_parent_evaluations is None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--conservative-rounds", type=int, default=4)
    parser.add_argument("--cascade-rounds", type=int, default=2)
    parser.add_argument("--max-hubs", type=int, default=64)
    parser.add_argument("--max-parent-evaluations", type=int, default=128)
    args = parser.parse_args()
    result = cascade_activation_star(
        TemporalGraph.from_json(args.input),
        args.root,
        conservative_rounds=args.conservative_rounds,
        cascade_rounds=args.cascade_rounds,
        max_hubs=args.max_hubs,
        max_parent_evaluations=args.max_parent_evaluations,
    )
    print(json.dumps(asdict(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
