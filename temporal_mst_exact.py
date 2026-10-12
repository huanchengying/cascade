#!/usr/bin/env python3
"""Bounded exact MST_w baselines; not a new polynomial-time MST algorithm.

Both backends use directed-Steiner subset DP. The reference backend uses the
existing explicit DAG; the implicit backend stores arrival channels and paid
arcs, without adjacency entries for zero-cost waiting. The terminal parameter
is the number of root-reachable physical vertices other than the root.
"""

from __future__ import annotations

import argparse
import json
import math
from array import array
from bisect import bisect_right
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from temporal_mst_prototype import (
    TemporalGraph,
    TimeStateDAG,
    _tree_from_edge_ids,
    earliest_arrival_tree,
    validate_temporal_tree,
)


Arc = Tuple[int, float, Optional[str]]
Decision = Optional[Tuple[str, int, Optional[str]]]


@dataclass
class ExactMSTResult:
    cost: float
    edge_ids: List[str]
    reachable: List[str]
    terminal_count: int
    state_count: int
    paid_arc_count: int
    explicit_zero_arc_count: int
    implicit_wait_count: int
    dp_cells: int
    backend: str


@dataclass
class _StateIndex:
    outgoing: List[List[Arc]]
    successor: List[int]
    reverse_order: List[int]
    terminal_bits: List[int]
    source: int
    reachable: List[str]


def _explicit_index(graph: TemporalGraph, root: str, reachable: List[str]) -> _StateIndex:
    dag = TimeStateDAG(graph, root)
    order = dag.topological_order()
    node_ids = {node: position for position, node in enumerate(order)}
    outgoing = [[] for _ in order]
    terminal_bits = [0] * len(order)
    terminals = [vertex for vertex in reachable if vertex != root]
    for position, vertex in enumerate(terminals):
        terminal_bits[node_ids[dag.terminals[vertex]]] = 1 << position
    for node, arcs in dag.adj.items():
        outgoing[node_ids[node]] = [
            (node_ids[target], cost, edge_id) for target, cost, edge_id in arcs
        ]
    return _StateIndex(
        outgoing, [-1] * len(order), list(reversed(range(len(order)))),
        terminal_bits, node_ids[dag.source], reachable,
    )


def _implicit_index(
    graph: TemporalGraph, root: str, reachable: List[str], earliest: Dict[str, float]
) -> _StateIndex:
    terminals = [vertex for vertex in reachable if vertex != root]
    reachable_set = set(reachable)
    bits = {vertex: 1 << position for position, vertex in enumerate(terminals)}
    eligible = [
        edge for edge in graph.edges
        if edge.v in bits and edge.u in reachable_set and earliest[edge.u] <= edge.start
        and edge.u != edge.v
    ]
    arrivals = {vertex: set() for vertex in terminals}
    for edge in eligible:
        arrivals[edge.v].add(edge.arrival)
    channels = {vertex: sorted(arrivals[vertex]) for vertex in terminals}
    state_ids = {}
    times = [-math.inf]
    terminal_bits = [0]
    for vertex in terminals:
        for timestamp in channels[vertex]:
            state_ids[vertex, timestamp] = len(times)
            times.append(timestamp)
            terminal_bits.append(bits[vertex])
    outgoing = [[] for _ in times]
    successor = [-1] * len(times)
    for vertex, timestamps in channels.items():
        for earlier, later in zip(timestamps, timestamps[1:]):
            successor[state_ids[vertex, earlier]] = state_ids[vertex, later]
    for edge in eligible:
        if edge.u == root:
            tail = 0
        else:
            timestamps = channels[edge.u]
            position = bisect_right(timestamps, edge.start) - 1
            if position < 0:
                continue
            tail = state_ids[edge.u, timestamps[position]]
        head = state_ids[edge.v, edge.arrival]
        outgoing[tail].append((head, edge.weight, edge.eid))
    return _StateIndex(
        outgoing, successor,
        sorted(range(len(times)), key=lambda node: times[node], reverse=True),
        terminal_bits, 0, reachable,
    )


def exact_mst_w(
    graph: TemporalGraph,
    root: str,
    *,
    backend: str = "implicit",
    max_terminals: int = 12,
    max_dp_cells: int = 2_000_000,
) -> ExactMSTResult:
    """Find an optimum tree spanning V_r, with explicit exponential-size guards.

    The root is available before every event, as in the existing prototype.
    Only positive-duration, finite nonnegative-weight inputs are supported.
    """
    if backend not in {"explicit", "channel_explicit", "implicit"}:
        raise ValueError("backend must be explicit, channel_explicit, or implicit")
    if any(edge.arrival <= edge.start for edge in graph.edges):
        raise ValueError("exact DAG baseline requires strictly positive durations")
    earliest, _ = earliest_arrival_tree(graph, root)
    reachable = [vertex for vertex in graph.vertices if earliest[vertex] < math.inf]
    terminal_count = len(reachable) - 1
    if terminal_count > max_terminals:
        raise ValueError(f"{terminal_count} terminals exceeds limit {max_terminals}")
    if max_dp_cells < 1:
        raise ValueError("max_dp_cells must be positive")
    if terminal_count == 0:
        return ExactMSTResult(0.0, [], reachable, 0, 1, 0, 0, 0, 0, backend)
    index = (
        _explicit_index(graph, root, reachable) if backend == "explicit"
        else _implicit_index(graph, root, reachable, earliest)
    )
    if backend == "channel_explicit":
        for node, successor in enumerate(index.successor):
            if successor >= 0:
                index.outgoing[node].append((successor, 0.0, None))
                index.successor[node] = -1
    state_count = len(index.outgoing)
    mask_count = 1 << terminal_count
    dp_cells = (mask_count - 1) * state_count
    if dp_cells > max_dp_cells:
        raise ValueError(f"{dp_cells} DP cells exceeds limit {max_dp_cells}")
    costs = [array("d")]
    decisions: List[List[Decision]] = [[]]
    for mask in range(1, mask_count):
        row = array("d", [math.inf]) * state_count
        choices: List[Decision] = [None] * state_count
        for node, terminal_bit in enumerate(index.terminal_bits):
            if terminal_bit == mask:
                row[node] = 0.0
        subset = (mask - 1) & mask
        while subset:
            complement = mask ^ subset
            if subset < complement:
                left, right = costs[subset], costs[complement]
                for node in range(state_count):
                    candidate = left[node] + right[node]
                    if candidate < row[node]:
                        row[node] = candidate
                        choices[node] = ("split", subset, None)
            subset = (subset - 1) & mask
        for node in index.reverse_order:
            successor = index.successor[node]
            if successor >= 0 and row[successor] < row[node]:
                row[node] = row[successor]
                choices[node] = ("step", successor, None)
            for target, weight, edge_id in index.outgoing[node]:
                candidate = weight + row[target]
                if candidate < row[node]:
                    row[node] = candidate
                    choices[node] = ("step", target, edge_id)
        costs.append(row)
        decisions.append(choices)
    optimum = costs[-1][index.source]
    if not math.isfinite(optimum):
        raise RuntimeError("reachable-terminal DP unexpectedly has no finite solution")
    selected = set()
    visited = set()
    pending = [(index.source, mask_count - 1)]
    while pending:
        node, mask = pending.pop()
        if (node, mask) in visited:
            continue
        visited.add((node, mask))
        decision = decisions[mask][node]
        if decision is None:
            if index.terminal_bits[node] != mask:
                raise RuntimeError("invalid DP reconstruction base case")
            continue
        kind, value, edge_id = decision
        if kind == "split":
            pending.extend(((node, value), (node, mask ^ value)))
        else:
            if edge_id is not None:
                selected.add(edge_id)
            pending.append((value, mask))
    tree = _tree_from_edge_ids(graph, root, sorted(selected), set(reachable))
    if not validate_temporal_tree(graph, root, tree, set(reachable)):
        raise RuntimeError("DP reconstruction is not a valid rooted temporal tree")
    tree_cost = math.fsum(edge.weight for edge in tree)
    if not math.isclose(tree_cost, optimum, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError(f"physical tree cost {tree_cost} disagrees with DP {optimum}")
    return ExactMSTResult(
        tree_cost, sorted(edge.eid for edge in tree), reachable, terminal_count,
        state_count,
        sum(edge_id is not None for arcs in index.outgoing for _, _, edge_id in arcs),
        sum(edge_id is None for arcs in index.outgoing for _, _, edge_id in arcs),
        sum(successor >= 0 for successor in index.successor), dp_cells, backend,
    )


def incoming_weight_lower_bound(graph: TemporalGraph, root: str) -> float:
    """Relax time coupling: every reachable non-root vertex needs one parent."""
    earliest, _ = earliest_arrival_tree(graph, root)
    minimum = {
        vertex: math.inf for vertex in graph.vertices
        if vertex != root and earliest[vertex] < math.inf
    }
    for edge in graph.edges:
        if edge.v in minimum and edge.u != edge.v and earliest[edge.u] <= edge.start:
            minimum[edge.v] = min(minimum[edge.v], edge.weight)
    return math.fsum(minimum.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--root", default="r")
    parser.add_argument("--backend", choices=["explicit", "channel_explicit", "implicit"], default="implicit")
    parser.add_argument("--max-terminals", type=int, default=12)
    parser.add_argument("--max-dp-cells", type=int, default=2_000_000)
    args = parser.parse_args()
    graph = TemporalGraph.from_json(args.input)
    result = exact_mst_w(
        graph, args.root, backend=args.backend, max_terminals=args.max_terminals,
        max_dp_cells=args.max_dp_cells,
    )
    payload = asdict(result)
    payload["incoming_weight_lower_bound"] = incoming_weight_lower_bound(graph, args.root)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
