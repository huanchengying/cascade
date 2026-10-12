#!/usr/bin/env python3
"""Deadline-constrained exact large-neighborhood search for temporal MST.

The current temporal tree fixes the arrival time of every outside vertex and
the departure deadlines of every fixed edge leaving a reoptimized region.  A
small region is detached by removing the parent edge of each region vertex.
All parent choices inside the region are then optimized jointly with the exact
implicit-DAG solver.  External tails are contracted into one already-reachable
anchor after checking their fixed arrival times.

Unlike a one-edge exchange, a neighborhood can deliberately buy an expensive
early edge into a hub and amortize it through several cheap descendants.  The
search is monotone because only validated, lower-cost temporal trees are
accepted.  It is a research prototype and should not be presented as novel
until the related-work audit is complete.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from temporal_mst_exact import ExactMSTResult, exact_mst_w
from temporal_mst_prototype import (
    TemporalEdge,
    TemporalGraph,
    temporal_reachable,
    validate_temporal_tree,
)
from temporal_mst_shared import (
    _pareto_hub_states,
    _shortest_temporal_paths,
    shortest_path_union_tree,
)


ANCHOR = "__fixed_tree_anchor__"


@dataclass
class SharedLNSResult:
    cost: float
    edge_ids: List[str]
    reachable: List[str]
    baseline_cost: float
    relative_improvement: float
    rounds: int
    accepted_neighborhoods: int
    evaluated_neighborhoods: int
    skipped_neighborhoods: int
    max_neighborhood: int
    algorithm: str = "deadline-shared-lns"


def _tree_state(
    root: str,
    tree: Sequence[TemporalEdge],
) -> Tuple[Dict[str, TemporalEdge], Dict[str, List[TemporalEdge]], Dict[str, float]]:
    parent = {edge.v: edge for edge in tree}
    children: Dict[str, List[TemporalEdge]] = defaultdict(list)
    for edge in tree:
        children[edge.u].append(edge)
    arrival = {root: -math.inf}
    arrival.update({vertex: edge.arrival for vertex, edge in parent.items()})
    return parent, children, arrival


def _is_descendant(
    candidate: str,
    ancestor: str,
    root: str,
    parent: Dict[str, TemporalEdge],
) -> bool:
    cursor = candidate
    while cursor != root and cursor in parent:
        if cursor == ancestor:
            return True
        cursor = parent[cursor].u
    return cursor == ancestor


def exact_region_reconnection(
    graph: TemporalGraph,
    root: str,
    tree: Sequence[TemporalEdge],
    region: Iterable[str],
    *,
    max_dp_cells: int = 4_000_000,
) -> Tuple[float, List[TemporalEdge], ExactMSTResult]:
    """Find the optimal replacement of all parent edges entering ``region``.

    Edges outside the region remain fixed.  An edge from a fixed outside vertex
    into the region is represented as an anchor edge after checking that the
    outside vertex reaches its departure time.  Fixed children outside the
    region induce arrival deadlines on their region parent.
    """

    required = temporal_reachable(graph, root, graph.edges)
    region_set = set(region) & required
    region_set.discard(root)
    if not region_set:
        raise ValueError("region must contain at least one non-root reachable vertex")
    parent, children, arrival = _tree_state(root, tree)
    if any(vertex not in parent for vertex in region_set):
        raise ValueError("region contains a vertex outside the current tree")
    deadlines = {
        vertex: min(
            (
                edge.start
                for edge in children.get(vertex, [])
                if edge.v not in region_set
            ),
            default=math.inf,
        )
        for vertex in region_set
    }

    local_edges: List[TemporalEdge] = []
    original_by_id = {edge.eid: edge for edge in graph.edges}
    for edge in graph.edges:
        if edge.v not in region_set or edge.u == edge.v:
            continue
        if edge.arrival > deadlines[edge.v]:
            continue
        if edge.u in region_set:
            local_edges.append(edge)
            continue
        if edge.u not in required or arrival.get(edge.u, math.inf) > edge.start:
            continue
        if _is_descendant(edge.u, edge.v, root, parent):
            continue
        local_edges.append(
            TemporalEdge(
                edge.eid,
                ANCHOR,
                edge.v,
                edge.start,
                edge.arrival,
                edge.weight,
            )
        )

    local_graph = TemporalGraph([ANCHOR] + sorted(region_set), local_edges)
    local_result = exact_mst_w(
        local_graph,
        ANCHOR,
        backend="implicit",
        max_terminals=len(region_set),
        max_dp_cells=max_dp_cells,
    )
    if set(local_result.reachable) != {ANCHOR} | region_set:
        raise RuntimeError("current tree should make every region vertex reconnectable")
    selected_ids = set(local_result.edge_ids)
    replacement = [original_by_id[edge_id] for edge_id in selected_ids]
    retained = [edge for edge in tree if edge.v not in region_set]
    candidate = retained + replacement
    if not validate_temporal_tree(graph, root, candidate, required):
        raise RuntimeError("exact region reconnection produced an invalid temporal tree")
    return math.fsum(edge.weight for edge in candidate), candidate, local_result


def _candidate_regions(
    graph: TemporalGraph,
    root: str,
    tree: Sequence[TemporalEdge],
    *,
    max_hub_states: int,
    max_states_per_vertex: int,
    max_neighborhood: int,
) -> List[FrozenSet[str]]:
    required = temporal_reachable(graph, root, graph.edges)
    parent = {edge.v: edge for edge in tree}
    edge_by_id = {edge.eid: edge for edge in graph.edges}
    root_index = _shortest_temporal_paths(graph, root)
    hubs = _pareto_hub_states(
        graph,
        root_index,
        max_states_per_vertex=max_states_per_vertex,
        max_hub_states=max_hub_states,
    )
    regions: List[FrozenSet[str]] = []
    seen = set()
    tree_ids = {item.eid for item in tree}
    marginal_costs = {
        edge.eid: (0.0 if edge.eid in tree_ids else edge.weight)
        for edge in graph.edges
    }
    for hub in hubs:
        suffix = _shortest_temporal_paths(
            graph,
            hub.vertex,
            source_time=hub.arrival,
            edge_costs=marginal_costs,
        )
        scored = []
        for terminal in required - {root, hub.vertex}:
            if terminal not in suffix.best_label:
                continue
            saving = parent[terminal].weight - suffix.best_cost[terminal]
            scored.append((saving, terminal))
        scored.sort(key=lambda item: (-item[0], item[1]))
        region = {hub.vertex}
        snapshots: List[FrozenSet[str]] = []
        for _, terminal in scored:
            path = suffix.path(terminal)
            path_vertices = {terminal}
            for edge_id in path:
                edge = edge_by_id[edge_id]
                if edge.u in required and edge.u != root:
                    path_vertices.add(edge.u)
                if edge.v in required and edge.v != root:
                    path_vertices.add(edge.v)
            if len(region | path_vertices) > max_neighborhood:
                continue
            region.update(path_vertices)
            if len(region) >= 2:
                snapshots.append(frozenset(region))
            if len(region) == max_neighborhood:
                break
        if snapshots:
            for snapshot in (snapshots[min(len(snapshots) - 1, 2)], snapshots[-1]):
                if snapshot not in seen:
                    seen.add(snapshot)
                    regions.append(snapshot)
    return regions


def shared_cost_lns(
    graph: TemporalGraph,
    root: str,
    *,
    max_rounds: int = 2,
    max_hub_states: int = 24,
    max_states_per_vertex: int = 4,
    max_neighborhood: int = 9,
    max_dp_cells: int = 4_000_000,
) -> SharedLNSResult:
    """Run monotone exact large-neighborhood search from a path-union tree."""

    if max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    if max_neighborhood < 2:
        raise ValueError("max_neighborhood must be at least two")
    required = temporal_reachable(graph, root, graph.edges)
    baseline_cost, baseline_tree, _ = shortest_path_union_tree(graph, root)
    current_cost = baseline_cost
    current_tree = list(baseline_tree)
    evaluated = 0
    skipped = 0
    accepted = 0
    completed_rounds = 0

    for round_number in range(1, max_rounds + 1):
        completed_rounds = round_number
        regions = _candidate_regions(
            graph,
            root,
            current_tree,
            max_hub_states=max_hub_states,
            max_states_per_vertex=max_states_per_vertex,
            max_neighborhood=max_neighborhood,
        )
        best_cost = current_cost
        best_tree: List[TemporalEdge] = []
        for region in regions:
            evaluated += 1
            try:
                candidate_cost, candidate_tree, _ = exact_region_reconnection(
                    graph,
                    root,
                    current_tree,
                    region,
                    max_dp_cells=max_dp_cells,
                )
            except (RuntimeError, ValueError):
                skipped += 1
                continue
            if candidate_cost + 1e-9 < best_cost:
                best_cost = candidate_cost
                best_tree = candidate_tree
        if not best_tree:
            break
        current_cost = best_cost
        current_tree = best_tree
        accepted += 1

    if not validate_temporal_tree(graph, root, current_tree, required):
        raise RuntimeError("shared-cost LNS returned an invalid tree")
    return SharedLNSResult(
        cost=current_cost,
        edge_ids=sorted(edge.eid for edge in current_tree),
        reachable=sorted(required),
        baseline_cost=baseline_cost,
        relative_improvement=(
            (baseline_cost - current_cost) / baseline_cost if baseline_cost > 0 else 0.0
        ),
        rounds=completed_rounds,
        accepted_neighborhoods=accepted,
        evaluated_neighborhoods=evaluated,
        skipped_neighborhoods=skipped,
        max_neighborhood=max_neighborhood,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--max-hub-states", type=int, default=24)
    parser.add_argument("--max-states-per-vertex", type=int, default=4)
    parser.add_argument("--max-neighborhood", type=int, default=9)
    parser.add_argument("--max-dp-cells", type=int, default=4_000_000)
    args = parser.parse_args()
    graph = TemporalGraph.from_json(args.input)
    result = shared_cost_lns(
        graph,
        args.root,
        max_rounds=args.max_rounds,
        max_hub_states=args.max_hub_states,
        max_states_per_vertex=args.max_states_per_vertex,
        max_neighborhood=args.max_neighborhood,
        max_dp_cells=args.max_dp_cells,
    )
    print(json.dumps(asdict(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
