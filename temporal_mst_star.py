#!/usr/bin/env python3
"""Exact shared-activation star neighborhoods for weighted temporal trees.

A one-edge descent cannot buy a more expensive early parent for a hub even
when that event unlocks many cheap outgoing events.  This module optimizes the
following deadline-preserving neighborhood exactly.  Replace the hub parent
once, then replace any subset of other parent events by events leaving that hub.
Every replacement must still meet the departure deadlines of all children in
the incumbent tree, even when another simultaneous replacement removes one of
those child edges.  This conservative deadline rule makes child replacements
independent after excluding the new ancestor path, so every negative marginal
replacement is selected.

The implementation sweeps candidate hub arrivals from late to early.  A
heavy-light decomposition maintains the total currently available child
saving and subtracts savings on the candidate tail's ancestor path, which are
the only replacements that would create a cycle.  This is a monotone primal
heuristic and a contribution candidate, not a novelty claim.  It is not exact
for the larger neighborhood that also exploits deadlines relaxed by the same
batch move.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from temporal_mst_prototype import TemporalEdge, TemporalGraph, temporal_reachable, validate_temporal_tree
from temporal_mst_shared import _parent_exchange_descent, shortest_path_union_tree


TOLERANCE = 1e-9


@dataclass
class SharedStarResult:
    cost: float
    edge_ids: List[str]
    reachable: List[str]
    baseline_cost: float
    exchange_cost: float
    relative_improvement: float
    rounds: int
    accepted_moves: int
    activated_children: int
    evaluated_hubs: int
    evaluated_parent_events: int
    algorithm: str = "exact-shared-activation-star"


@dataclass
class _StarDescentResult:
    cost: float
    tree: List[TemporalEdge]
    rounds: int
    accepted_moves: int
    activated_children: int
    evaluated_hubs: int
    evaluated_parent_events: int


class _Fenwick:
    def __init__(self, size: int):
        self.size = size
        self.values: Dict[int, float] = {}

    def add(self, position: int, value: float) -> None:
        position += 1
        while position <= self.size:
            self.values[position] = self.values.get(position, 0.0) + value
            position += position & -position

    def prefix_sum(self, stop: int) -> float:
        result = 0.0
        while stop > 0:
            result += self.values.get(stop, 0.0)
            stop -= stop & -stop
        return result

    def range_sum(self, start: int, stop: int) -> float:
        return self.prefix_sum(stop) - self.prefix_sum(start)


class _TreeIndex:
    def __init__(self, root: str, tree: Sequence[TemporalEdge], required: Set[str]):
        self.root = root
        self.parent_edge = {edge.v: edge for edge in tree}
        if set(self.parent_edge) != required - {root}:
            raise ValueError("tree must contain one parent for every non-root vertex")
        self.parent = {root: root}
        self.children: Dict[str, List[str]] = {vertex: [] for vertex in required}
        for edge in tree:
            self.parent[edge.v] = edge.u
            self.children[edge.u].append(edge.v)
        for children in self.children.values():
            children.sort()

        self.depth = {root: 0}
        self.tin: Dict[str, int] = {}
        self.tout: Dict[str, int] = {}
        order: List[str] = []
        stack: List[Tuple[str, bool]] = [(root, False)]
        timer = 0
        while stack:
            vertex, leaving = stack.pop()
            if leaving:
                self.tout[vertex] = timer
                continue
            self.tin[vertex] = timer
            timer += 1
            order.append(vertex)
            stack.append((vertex, True))
            for child in reversed(self.children[vertex]):
                self.depth[child] = self.depth[vertex] + 1
                stack.append((child, False))
        if len(order) != len(required):
            raise ValueError("selected parents do not form a rooted tree")
        self.postorder = list(reversed(order))
        self.children_by_start = {
            vertex: sorted(
                self.children[vertex],
                key=lambda child: (self.parent_edge[child].start, child),
            )
            for vertex in required
        }

        size = {vertex: 1 for vertex in required}
        heavy: Dict[str, Optional[str]] = {vertex: None for vertex in required}
        for vertex in reversed(order):
            best_size = 0
            for child in self.children[vertex]:
                size[vertex] += size[child]
                if size[child] > best_size:
                    best_size = size[child]
                    heavy[vertex] = child

        self.head: Dict[str, str] = {}
        self.position: Dict[str, int] = {}
        position = 0
        pending = [(root, root)]
        while pending:
            vertex, chain_head = pending.pop()
            cursor: Optional[str] = vertex
            while cursor is not None:
                self.head[cursor] = chain_head
                self.position[cursor] = position
                position += 1
                light = [child for child in self.children[cursor] if child != heavy[cursor]]
                for child in reversed(light):
                    pending.append((child, child))
                cursor = heavy[cursor]

        self.arrival = {root: -math.inf}
        self.arrival.update({vertex: edge.arrival for vertex, edge in self.parent_edge.items()})
        self.deadline = {
            vertex: min(
                (self.parent_edge[child].start for child in self.children[vertex]),
                default=math.inf,
            )
            for vertex in required
        }

    def is_ancestor(self, ancestor: str, vertex: str) -> bool:
        return self.tin[ancestor] <= self.tin[vertex] < self.tout[ancestor]

    def root_path_sum(self, vertex: str, fenwick: _Fenwick) -> float:
        result = 0.0
        while self.head[vertex] != self.head[self.root]:
            chain_head = self.head[vertex]
            result += fenwick.range_sum(
                self.position[chain_head], self.position[vertex] + 1
            )
            vertex = self.parent[chain_head]
        result += fenwick.range_sum(
            self.position[self.root], self.position[vertex] + 1
        )
        return result


@dataclass(frozen=True)
class _Move:
    saving: float
    hub: str
    parent_edge: TemporalEdge


def _candidate_groups(
    graph: TemporalGraph,
    required: Set[str],
) -> Tuple[Dict[str, List[TemporalEdge]], Dict[str, List[TemporalEdge]]]:
    incoming: Dict[str, List[TemporalEdge]] = defaultdict(list)
    outgoing: Dict[str, List[TemporalEdge]] = defaultdict(list)
    for edge in graph.edges:
        if (
            edge.u in required
            and edge.v in required
            and edge.v != edge.u
        ):
            incoming[edge.v].append(edge)
            outgoing[edge.u].append(edge)
    for edges in incoming.values():
        edges.sort(key=lambda edge: (-edge.arrival, edge.weight, edge.eid))
    for edges in outgoing.values():
        edges.sort(key=lambda edge: (-edge.start, edge.weight, edge.arrival, edge.eid))
    return incoming, outgoing


def _best_move(
    graph: TemporalGraph,
    root: str,
    required: Set[str],
    tree: Sequence[TemporalEdge],
) -> Tuple[Optional[_Move], int, int]:
    index = _TreeIndex(root, tree, required)
    incoming, outgoing = _candidate_groups(graph, required)
    best_move: Optional[_Move] = None
    evaluated_hubs = 0
    evaluated_parents = 0

    for hub in sorted(required - {root}):
        parent_candidates = [
            edge
            for edge in incoming.get(hub, [])
            if index.arrival[edge.u] <= edge.start
            and not index.is_ancestor(hub, edge.u)
            and edge.arrival <= index.deadline[hub]
        ]
        if not parent_candidates:
            continue
        child_candidates = [
            edge
            for edge in outgoing.get(hub, [])
            if edge.v not in {root, hub}
            and edge.arrival <= index.deadline[edge.v]
        ]
        if not child_candidates:
            continue
        evaluated_hubs += 1
        fenwick = _Fenwick(len(required))
        best_child: Dict[str, TemporalEdge] = {}
        gain: Dict[str, float] = {}
        total_gain = 0.0
        child_offset = 0

        for parent_edge in parent_candidates:
            while (
                child_offset < len(child_candidates)
                and child_candidates[child_offset].start >= parent_edge.arrival
            ):
                edge = child_candidates[child_offset]
                child_offset += 1
                previous = best_child.get(edge.v)
                if previous is not None and (
                    previous.weight,
                    previous.arrival,
                    previous.eid,
                ) <= (edge.weight, edge.arrival, edge.eid):
                    continue
                old_gain = gain.get(edge.v, 0.0)
                best_child[edge.v] = edge
                new_gain = max(0.0, index.parent_edge[edge.v].weight - edge.weight)
                gain[edge.v] = new_gain
                delta = new_gain - old_gain
                total_gain += delta
                fenwick.add(index.position[edge.v], delta)

            evaluated_parents += 1
            cycle_gain = index.root_path_sum(parent_edge.u, fenwick)
            saving = (
                index.parent_edge[hub].weight
                - parent_edge.weight
                + total_gain
                - cycle_gain
            )
            candidate = _Move(saving, hub, parent_edge)
            if saving <= TOLERANCE:
                continue
            if best_move is None or (
                -candidate.saving,
                candidate.hub,
                candidate.parent_edge.eid,
            ) < (
                -best_move.saving,
                best_move.hub,
                best_move.parent_edge.eid,
            ):
                best_move = candidate

    return best_move, evaluated_hubs, evaluated_parents


def _apply_move(
    graph: TemporalGraph,
    root: str,
    required: Set[str],
    tree: Sequence[TemporalEdge],
    move: _Move,
) -> Tuple[float, List[TemporalEdge], int]:
    index = _TreeIndex(root, tree, required)
    best_child: Dict[str, TemporalEdge] = {}
    for edge in graph.edges:
        if (
            edge.u != move.hub
            or edge.v not in required
            or edge.v in {root, move.hub}
            or edge.start < move.parent_edge.arrival
            or edge.arrival > index.deadline[edge.v]
            or index.is_ancestor(edge.v, move.parent_edge.u)
        ):
            continue
        previous = best_child.get(edge.v)
        if previous is None or (
            edge.weight,
            edge.arrival,
            edge.eid,
        ) < (previous.weight, previous.arrival, previous.eid):
            best_child[edge.v] = edge

    parent = dict(index.parent_edge)
    parent[move.hub] = move.parent_edge
    activated = 0
    for vertex, edge in best_child.items():
        if edge.weight + TOLERANCE < parent[vertex].weight:
            parent[vertex] = edge
            activated += 1
    candidate = list(parent.values())
    if not validate_temporal_tree(graph, root, candidate, required):
        raise RuntimeError("shared-activation move failed physical-tree validation")
    old_cost = math.fsum(edge.weight for edge in tree)
    cost = math.fsum(edge.weight for edge in candidate)
    if not math.isclose(old_cost - cost, move.saving, rel_tol=1e-8, abs_tol=1e-7):
        raise RuntimeError("shared-activation sweep and reconstructed move disagree")
    return cost, candidate, activated


def _shared_activation_descent(
    graph: TemporalGraph,
    root: str,
    required: Set[str],
    initial_tree: Sequence[TemporalEdge],
    *,
    max_rounds: int,
) -> _StarDescentResult:
    current_tree = list(initial_tree)
    current_cost = math.fsum(edge.weight for edge in current_tree)
    accepted = 0
    activated_children = 0
    evaluated_hubs = 0
    evaluated_parents = 0
    rounds = 0
    for round_number in range(1, max_rounds + 1):
        rounds = round_number
        move, hubs, parents = _best_move(
            graph, root, required, current_tree
        )
        evaluated_hubs += hubs
        evaluated_parents += parents
        if move is None:
            break
        candidate_cost, candidate, activated = _apply_move(
            graph, root, required, current_tree, move
        )
        if candidate_cost + TOLERANCE >= current_cost:
            raise RuntimeError("accepted shared-activation move is not improving")
        current_cost = candidate_cost
        current_tree = candidate
        accepted += 1
        activated_children += activated
    return _StarDescentResult(
        current_cost,
        current_tree,
        rounds,
        accepted,
        activated_children,
        evaluated_hubs,
        evaluated_parents,
    )


def shared_activation_star(
    graph: TemporalGraph,
    root: str,
    *,
    max_rounds: int = 8,
) -> SharedStarResult:
    """Run exact best-improvement shared-activation star descent."""

    if max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    if any(edge.arrival <= edge.start for edge in graph.edges):
        raise ValueError("shared-activation stars require positive durations")
    required = temporal_reachable(graph, root, graph.edges)
    baseline_cost, baseline_tree, _ = shortest_path_union_tree(graph, root)
    current_tree, _ = _parent_exchange_descent(
        graph, root, required, baseline_tree
    )
    exchange_cost = math.fsum(edge.weight for edge in current_tree)
    descent = _shared_activation_descent(
        graph,
        root,
        required,
        current_tree,
        max_rounds=max_rounds,
    )
    current_cost = descent.cost
    current_tree = descent.tree

    if not validate_temporal_tree(graph, root, current_tree, required):
        raise RuntimeError("shared-activation descent returned an invalid tree")
    return SharedStarResult(
        cost=current_cost,
        edge_ids=sorted(edge.eid for edge in current_tree),
        reachable=sorted(required),
        baseline_cost=baseline_cost,
        exchange_cost=exchange_cost,
        relative_improvement=(
            (baseline_cost - current_cost) / baseline_cost if baseline_cost else 0.0
        ),
        rounds=descent.rounds,
        accepted_moves=descent.accepted_moves,
        activated_children=descent.activated_children,
        evaluated_hubs=descent.evaluated_hubs,
        evaluated_parent_events=descent.evaluated_parent_events,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--max-rounds", type=int, default=8)
    args = parser.parse_args()
    graph = TemporalGraph.from_json(args.input)
    result = shared_activation_star(graph, args.root, max_rounds=args.max_rounds)
    print(json.dumps(asdict(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
