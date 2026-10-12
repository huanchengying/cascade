#!/usr/bin/env python3
"""Shared-availability Lagrangian bounds and primal recovery for ``MST_w``.

For strictly positive durations, Ikuta's temporal constraints and one-incoming
edge constraints already exclude directed cycles: arrival times strictly
increase along every selected parent edge.  We relax only the temporal support
constraints.  Their multipliers penalize unsupported outgoing edges and give
an aggregate credit to an earlier incoming edge for every downstream edge it
can enable.  The relaxed problem then decomposes into one minimum adjusted-cost
incoming edge per vertex.

The best dual value is a certified lower bound.  Reduced-cost temporal path
unions plus deadline-safe parent exchanges provide feasible upper bounds.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from temporal_mst_ilp import _candidate_edges, ikuta_ilp
from temporal_mst_prototype import (
    TemporalEdge,
    TemporalGraph,
    _tree_from_edge_ids,
    temporal_reachable,
    validate_temporal_tree,
)
from temporal_mst_shared import (
    _parent_exchange_descent,
    _shortest_temporal_paths,
    shortest_path_union_tree,
)


@dataclass
class LagrangianResult:
    cost: float
    edge_ids: List[str]
    reachable: List[str]
    baseline_cost: float
    lower_bound: float
    relative_gap: float
    iterations: int
    feasible_relaxed_solutions: int
    recovered_solutions: int
    original_edge_count: int
    pool_edge_count: int
    pool_ilp_status: str
    safe_edge_count: int
    safe_reduction_ratio: float
    safe_master_status: str
    certified_optimal: bool
    bound_trace: List[Tuple[int, float, float]]
    algorithm: str = "shared-availability-lagrangian"


def _selected_tree(
    graph: TemporalGraph,
    root: str,
    required: Set[str],
    edges: Sequence[TemporalEdge],
    chosen: Sequence[int],
) -> Optional[List[TemporalEdge]]:
    selected = [edges[position] for position in chosen]
    return selected if validate_temporal_tree(graph, root, selected, required) else None


def _reduced_cost_tree(
    graph: TemporalGraph,
    root: str,
    required: Set[str],
    adjusted: np.ndarray,
    edges: Sequence[TemporalEdge],
) -> Optional[List[TemporalEdge]]:
    work_graph = TemporalGraph(list(graph.vertices), list(edges))
    costs = {edge.eid: float(adjusted[position]) for position, edge in enumerate(edges)}
    index = _shortest_temporal_paths(work_graph, root, edge_costs=costs)
    paths = []
    for terminal in required - {root}:
        if terminal not in index.best_label:
            return None
        paths.append(index.path(terminal))
    tree = _tree_from_edge_ids(
        graph,
        root,
        {edge_id for path in paths for edge_id in path},
        required,
    )
    return tree if validate_temporal_tree(graph, root, tree, required) else None


def shared_availability_lagrangian(
    graph: TemporalGraph,
    root: str,
    *,
    iterations: int = 300,
    primal_interval: int = 10,
    step_scale: float = 1.8,
    tolerance: float = 1e-6,
    pool_width: int = 4,
    pool_interval: int = 10,
    pool_time_limit: Optional[float] = 30.0,
    safe_master_time_limit: Optional[float] = 0.0,
) -> LagrangianResult:
    """Optimize a Lagrangian lower bound and recover feasible temporal trees."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if primal_interval < 1:
        raise ValueError("primal_interval must be positive")
    if step_scale <= 0:
        raise ValueError("step_scale must be positive")
    if pool_width < 0:
        raise ValueError("pool_width must be non-negative")
    if pool_interval < 1:
        raise ValueError("pool_interval must be positive")
    if pool_time_limit is not None and pool_time_limit < 0:
        raise ValueError("pool_time_limit must be non-negative")
    if safe_master_time_limit is not None and safe_master_time_limit < 0:
        raise ValueError("safe_master_time_limit must be non-negative")
    if any(edge.arrival <= edge.start for edge in graph.edges):
        raise ValueError("Lagrangian cut elimination requires positive durations")

    reachable, edges = _candidate_edges(graph, root)
    required = set(reachable)
    if len(required) == 1:
        return LagrangianResult(
            0.0, [], reachable, 0.0, 0.0, 0.0, 0, 1, 1, 0, 0,
            "not-needed", 0, 0.0, "not-needed", True,
            [(0, 0.0, 0.0)]
        )
    work_graph = TemporalGraph(list(graph.vertices), list(edges))
    edge_count = len(edges)
    incoming: Dict[str, List[int]] = {vertex: [] for vertex in reachable}
    outgoing: Dict[str, List[int]] = {vertex: [] for vertex in reachable}
    for position, edge in enumerate(edges):
        incoming[edge.v].append(position)
        outgoing[edge.u].append(position)
    for vertex in reachable:
        incoming[vertex].sort(key=lambda position: (edges[position].arrival, position))
        outgoing[vertex].sort(key=lambda position: (edges[position].start, position))
    outgoing_starts = {
        vertex: [edges[position].start for position in positions]
        for vertex, positions in outgoing.items()
    }

    baseline_cost, baseline_tree, _ = shortest_path_union_tree(work_graph, root)
    best_tree, _ = _parent_exchange_descent(
        graph, root, required, baseline_tree
    )
    best_upper = math.fsum(edge.weight for edge in best_tree)
    multipliers = np.zeros(edge_count, dtype=float)
    constrained = np.array([edge.u != root for edge in edges], dtype=bool)
    best_lower = -math.inf
    best_adjusted: Optional[np.ndarray] = None
    feasible_relaxed = 0
    recovered = 1
    trace: List[Tuple[int, float, float]] = []
    pool_edge_ids = {
        edge.eid for edge in [*baseline_tree, *best_tree]
    }
    stagnant = 0
    theta = step_scale

    for iteration in range(1, iterations + 1):
        adjusted = np.array([edge.weight for edge in edges], dtype=float)
        adjusted[constrained] += multipliers[constrained]
        for vertex in reachable:
            outgoing_positions = outgoing[vertex]
            if not outgoing_positions:
                continue
            starts = outgoing_starts[vertex]
            suffix = np.zeros(len(outgoing_positions) + 1, dtype=float)
            for offset in range(len(outgoing_positions) - 1, -1, -1):
                suffix[offset] = suffix[offset + 1] + multipliers[outgoing_positions[offset]]
            offset = 0
            for incoming_position in incoming[vertex]:
                arrival = edges[incoming_position].arrival
                while offset < len(starts) and starts[offset] < arrival:
                    offset += 1
                adjusted[incoming_position] -= suffix[offset]

        chosen: List[int] = []
        chosen_by_head: Dict[str, int] = {}
        dual = 0.0
        for vertex in reachable:
            if vertex == root:
                continue
            key = lambda item: (
                adjusted[item], edges[item].weight, edges[item].arrival, item
            )
            collect_pool = bool(
                pool_width and (iteration == 1 or iteration % pool_interval == 0)
            )
            if collect_pool:
                ranked = heapq.nsmallest(pool_width, incoming[vertex], key=key)
                position = ranked[0]
            else:
                ranked = []
                position = min(incoming[vertex], key=key)
            chosen.append(position)
            chosen_by_head[vertex] = position
            dual += adjusted[position]
            pool_edge_ids.add(edges[position].eid)
            if collect_pool:
                pool_edge_ids.update(
                    edges[item].eid for item in ranked
                )

        if dual > best_lower + tolerance:
            best_lower = dual
            best_adjusted = adjusted.copy()
            stagnant = 0
        else:
            stagnant += 1
        if iteration == 1 or iteration % primal_interval == 0:
            trace.append((iteration, float(best_lower), float(best_upper)))

        selected_tree = _selected_tree(graph, root, required, edges, chosen)
        if selected_tree is not None:
            feasible_relaxed += 1
            candidate, _ = _parent_exchange_descent(
                graph, root, required, selected_tree
            )
            candidate_cost = math.fsum(edge.weight for edge in candidate)
            if candidate_cost < best_upper:
                best_upper = candidate_cost
                best_tree = candidate

        if iteration % primal_interval == 0 or iteration == iterations:
            candidate = _reduced_cost_tree(
                graph, root, required, adjusted, edges
            )
            if candidate is not None:
                recovered += 1
                pool_edge_ids.update(edge.eid for edge in candidate)
                candidate, _ = _parent_exchange_descent(
                    graph, root, required, candidate
                )
                candidate_cost = math.fsum(edge.weight for edge in candidate)
                if candidate_cost < best_upper:
                    best_upper = candidate_cost
                    best_tree = candidate

        selected_arrival = {
            vertex: edges[position].arrival
            for vertex, position in chosen_by_head.items()
        }
        selected_mask = np.zeros(edge_count, dtype=float)
        selected_mask[chosen] = 1.0
        selected_mask[~constrained] = 0.0
        gradient = selected_mask
        for position, edge in enumerate(edges):
            if edge.u == root:
                continue
            support = float(selected_arrival[edge.u] <= edge.start)
            gradient[position] -= support
        norm_squared = float(np.dot(gradient, gradient))
        if norm_squared <= tolerance:
            break
        dual_gap = max(0.0, best_upper - dual)
        if dual_gap <= tolerance:
            best_lower = min(best_upper, max(best_lower, dual))
            break
        step = theta * dual_gap / norm_squared
        multipliers = np.maximum(0.0, multipliers + step * gradient)
        if stagnant >= 25:
            theta = max(0.05, theta * 0.5)
            stagnant = 0

    if best_adjusted is not None:
        if pool_width:
            for vertex in reachable:
                if vertex == root:
                    continue
                key = lambda item: (
                    best_adjusted[item],
                    edges[item].weight,
                    edges[item].arrival,
                    item,
                )
                pool_edge_ids.update(
                    edges[item].eid
                    for item in heapq.nsmallest(pool_width, incoming[vertex], key=key)
                )
        candidate = _reduced_cost_tree(
            graph, root, required, best_adjusted, edges
        )
        if candidate is not None:
            recovered += 1
            pool_edge_ids.update(edge.eid for edge in candidate)
            candidate, _ = _parent_exchange_descent(
                graph, root, required, candidate
            )
            candidate_cost = math.fsum(edge.weight for edge in candidate)
            if candidate_cost < best_upper:
                best_upper = candidate_cost
                best_tree = candidate
    pool_status = "disabled"
    if pool_width and pool_time_limit != 0:
        pooled_edges = [edge for edge in edges if edge.eid in pool_edge_ids]
        pooled_graph = TemporalGraph(list(graph.vertices), pooled_edges)
        try:
            pool_result = ikuta_ilp(
                pooled_graph,
                root,
                time_limit=pool_time_limit,
            )
            pool_status = pool_result.status
            pool_tree = [
                edge for edge in graph.edges if edge.eid in set(pool_result.edge_ids)
            ]
            if validate_temporal_tree(graph, root, pool_tree, required):
                pool_cost = math.fsum(edge.weight for edge in pool_tree)
                if pool_cost < best_upper:
                    best_upper = pool_cost
                    best_tree = pool_tree
        except (RuntimeError, TimeoutError) as exc:
            pool_status = f"failed: {type(exc).__name__}"
    safe_edge_ids = set(edge.eid for edge in best_tree)
    safe_status = "disabled"
    if best_adjusted is not None:
        reduced_cost_budget = max(0.0, best_upper - best_lower)
        minimum_adjusted = {
            vertex: min(float(best_adjusted[position]) for position in positions)
            for vertex, positions in incoming.items()
            if vertex != root
        }
        for position, edge in enumerate(edges):
            excess = float(best_adjusted[position]) - minimum_adjusted[edge.v]
            if excess <= reduced_cost_budget + tolerance:
                safe_edge_ids.add(edge.eid)
        if safe_master_time_limit != 0:
            safe_graph = TemporalGraph(
                list(graph.vertices),
                [edge for edge in edges if edge.eid in safe_edge_ids],
            )
            try:
                safe_result = ikuta_ilp(
                    safe_graph,
                    root,
                    time_limit=safe_master_time_limit,
                )
                safe_status = safe_result.status
                best_lower = max(best_lower, safe_result.lower_bound)
                safe_tree = [
                    edge for edge in graph.edges if edge.eid in set(safe_result.edge_ids)
                ]
                if validate_temporal_tree(graph, root, safe_tree, required):
                    safe_cost = math.fsum(edge.weight for edge in safe_tree)
                    if safe_cost < best_upper:
                        best_upper = safe_cost
                        best_tree = safe_tree
                safe_optimal = safe_result.optimal
            except (RuntimeError, TimeoutError) as exc:
                safe_status = f"failed: {type(exc).__name__}"
                safe_optimal = False
        else:
            safe_status = "not-solved"
            safe_optimal = False
    else:
        safe_optimal = False
    if not validate_temporal_tree(graph, root, best_tree, required):
        raise RuntimeError("Lagrangian primal recovery returned an invalid tree")
    best_lower = float(min(best_lower, best_upper))
    relative_gap = (
        (best_upper - best_lower) / best_upper if best_upper > 0 else 0.0
    )
    if not trace or trace[-1][0] != iteration:
        trace.append((iteration, float(best_lower), float(best_upper)))
    return LagrangianResult(
        cost=best_upper,
        edge_ids=sorted(edge.eid for edge in best_tree),
        reachable=reachable,
        baseline_cost=baseline_cost,
        lower_bound=best_lower,
        relative_gap=relative_gap,
        iterations=iteration,
        feasible_relaxed_solutions=feasible_relaxed,
        recovered_solutions=recovered,
        original_edge_count=edge_count,
        pool_edge_count=len(pool_edge_ids),
        pool_ilp_status=pool_status,
        safe_edge_count=len(safe_edge_ids),
        safe_reduction_ratio=(
            1.0 - len(safe_edge_ids) / edge_count if edge_count else 0.0
        ),
        safe_master_status=safe_status,
        certified_optimal=safe_optimal or relative_gap <= tolerance,
        bound_trace=trace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--primal-interval", type=int, default=10)
    parser.add_argument("--step-scale", type=float, default=1.8)
    parser.add_argument("--pool-width", type=int, default=4)
    parser.add_argument("--pool-interval", type=int, default=10)
    parser.add_argument("--pool-time-limit", type=float, default=30.0)
    parser.add_argument("--safe-master-time-limit", type=float, default=0.0)
    args = parser.parse_args()
    graph = TemporalGraph.from_json(args.input)
    result = shared_availability_lagrangian(
        graph,
        args.root,
        iterations=args.iterations,
        primal_interval=args.primal_interval,
        step_scale=args.step_scale,
        pool_width=args.pool_width,
        pool_interval=args.pool_interval,
        pool_time_limit=args.pool_time_limit,
        safe_master_time_limit=args.safe_master_time_limit,
    )
    print(json.dumps(asdict(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
