#!/usr/bin/env python3
"""Huang--Fu--Liu Algorithms 4/5 on the explicit metric closure.

This is a Python reference implementation with size/time guards, not the
authors' C++ binary. The improved recursive density search is implemented;
the optional Algorithm 6 lazy vertex-order enhancement is not implemented.
No approximation guarantee is claimed for interrupted or guarded-out runs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Set, Tuple

from temporal_mst_prototype import TemporalGraph, TimeStateDAG, _tree_from_edge_ids, validate_temporal_tree


@dataclass(frozen=True)
class _ClosureTree:
    arcs: FrozenSet[Tuple[int, int]]
    terminals: FrozenSet[int]


@dataclass
class HuangResult:
    cost: float
    edge_ids: List[str]
    reachable: List[str]
    level: int
    state_count: int
    closure_arcs: int
    recursive_calls: int
    closure_cost: float
    wall_time_s: float
    algorithm: str = "huang-algorithms-4-5"


def huang_recursive_greedy(graph: TemporalGraph, root: str, *, level: int = 2, time_limit: float = 30.0, max_states: int = 400, max_terminals: int = 50, search: str = "improved") -> HuangResult:
    if level not in {1, 2, 3} or time_limit <= 0 or max_states < 1 or max_terminals < 1:
        raise ValueError("invalid Huang reference settings")
    if search not in {"improved", "naive"}:
        raise ValueError("search must be improved or naive")
    if any(edge.arrival <= edge.start for edge in graph.edges):
        raise ValueError("Huang reference DAG requires positive durations")
    started = time.monotonic()

    def check_budget():
        if time.monotonic() - started >= time_limit:
            raise TimeoutError("Huang recursive density search exceeded time limit")

    dag = TimeStateDAG(graph, root)
    order = sorted(dag.adj, key=lambda node: (node != dag.source, node))
    topological = dag.topological_order()
    node_id = {node: position for position, node in enumerate(order)}
    terminals = frozenset(node_id[node] for node in dag.terminals.values())
    if len(order) > max_states or len(terminals) > max_terminals:
        raise ValueError(f"Huang metric closure guard: {len(order)} states, {len(terminals)} terminals")
    distances = []
    predecessors = []
    for source in order:
        check_budget()
        distance = {source: 0.0}
        previous = {}
        for tail in topological:
            if tail not in distance:
                continue
            for head, cost, physical in dag.adj[tail]:
                candidate = distance[tail] + cost
                if candidate < distance.get(head, math.inf):
                    distance[head] = candidate
                    previous[node_id[head]] = (node_id[tail], physical)
        distances.append({node_id[node]: cost for node, cost in distance.items()})
        predecessors.append(previous)
    calls = 0

    def cost(tree: _ClosureTree) -> float:
        return math.fsum(distances[tail][head] for tail, head in tree.arcs)

    def attach(tree: _ClosureTree, tail: int, head: int) -> _ClosureTree:
        return _ClosureTree(tree.arcs if tail == head else tree.arcs | {(tail, head)}, tree.terminals)

    def merge(left: _ClosureTree, right: _ClosureTree) -> _ClosureTree:
        return _ClosureTree(left.arcs | right.arcs, left.terminals | right.terminals)

    @lru_cache(maxsize=50_000)
    def branches(height: int, source: int, target_count: int, targets: FrozenSet[int], incoming_cost: float) -> _ClosureTree:
        nonlocal calls
        calls += 1
        check_budget()
        current = _ClosureTree(frozenset(), frozenset())
        best = current
        best_density = math.inf
        remaining = set(targets)
        count = target_count
        while count > 0:
            check_budget()
            if height == 1:
                available = [target for target in remaining if target in distances[source]]
                if not available:
                    break
                terminal = min(available, key=lambda target: (distances[source][target], target))
                candidate = _ClosureTree(frozenset({(source, terminal)}) if source != terminal else frozenset(), frozenset({terminal}))
            else:
                candidate = best_branch(height, source, count, frozenset(remaining))
                if not candidate.terminals:
                    break
            current = merge(current, candidate)
            newly = remaining & candidate.terminals
            count -= len(newly)
            remaining -= newly
            density = (cost(current) + incoming_cost) / len(current.terminals)
            if density < best_density:
                best, best_density = current, density
        return best

    def best_branch(height: int, source: int, count: int, targets: FrozenSet[int]) -> _ClosureTree:
        best = _ClosureTree(frozenset(), frozenset())
        best_density = math.inf
        for hub in sorted(distances[source]):
            check_budget()
            if not any(target in distances[hub] for target in targets):
                continue
            if search == "naive":
                options = [full_tree(height - 1, hub, requested, targets) for requested in range(1, count + 1)]
                options = [attach(option, source, hub) for option in options if option.terminals]
                if not options:
                    continue
                candidate = min(options, key=lambda option: cost(option) / len(option.terminals))
            else:
                candidate = branches(height - 1, hub, count, targets, distances[source][hub])
            if not candidate.terminals:
                continue
            candidate = attach(candidate, source, hub)
            density = cost(candidate) / len(candidate.terminals)
            if density < best_density:
                best, best_density = candidate, density
        return best

    @lru_cache(maxsize=50_000)
    def full_tree(height: int, source: int, count: int, targets: FrozenSet[int]) -> _ClosureTree:
        nonlocal calls
        calls += 1
        current = _ClosureTree(frozenset(), frozenset())
        remaining = set(targets)
        while count > 0:
            check_budget()
            if height == 1:
                available = [target for target in remaining if target in distances[source]]
                if not available:
                    break
                target = min(available, key=lambda node: (distances[source][node], node))
                candidate = _ClosureTree(frozenset({(source, target)}) if source != target else frozenset(), frozenset({target}))
            else:
                candidate = best_branch(height, source, count, frozenset(remaining))
                if not candidate.terminals:
                    break
            current = merge(current, candidate)
            newly = remaining & candidate.terminals
            count -= len(newly)
            remaining -= newly
        return current

    current = _ClosureTree(frozenset(), frozenset())
    remaining = terminals
    source = node_id[dag.source]
    while remaining:
        check_budget()
        if level == 1:
            target = min(remaining, key=lambda node: (distances[source][node], node))
            candidate = _ClosureTree(frozenset({(source, target)}), frozenset({target}))
        else:
            candidate = best_branch(level, source, len(remaining), remaining)
        if not candidate.terminals:
            raise RuntimeError("Huang density search failed to cover reachable terminals")
        current = merge(current, candidate)
        remaining = remaining - candidate.terminals
    expanded: Set[str] = set()
    for tail, head in current.arcs:
        cursor = head
        while cursor != tail:
            predecessor, physical = predecessors[tail][cursor]
            if physical is not None:
                expanded.add(physical)
            cursor = predecessor
    tree = _tree_from_edge_ids(graph, root, sorted(expanded), dag.reachable)
    if not validate_temporal_tree(graph, root, tree, dag.reachable):
        raise RuntimeError("Huang expansion is not a feasible temporal tree")
    return HuangResult(math.fsum(edge.weight for edge in tree), sorted(edge.eid for edge in tree), sorted(dag.reachable), level, len(order), sum(len(row) - 1 for row in distances), calls, cost(current), time.monotonic() - started, "huang-algorithm-3" if search == "naive" else "huang-algorithms-4-5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=30)
    args = parser.parse_args()
    print(json.dumps(asdict(huang_recursive_greedy(TemporalGraph.from_json(args.input), args.root, level=args.level, time_limit=args.time_limit)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
