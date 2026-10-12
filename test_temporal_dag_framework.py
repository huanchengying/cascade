#!/usr/bin/env python3
"""Random differential tests for the compact temporal DAG applications."""

from __future__ import annotations

import math
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

from temporal_dag_framework import (
    CompactTemporalDAG,
    fastest_path,
    mst_a,
    shortest_path,
    mst_w_upper_bound,
)


def brute_force(
    src: List[int],
    dst: List[int],
    start: List[int],
    arrival: List[int],
    weight: List[float],
    vertex_count: int,
    source: int,
    source_time: int,
) -> Tuple[List[int], List[float], List[float]]:
    """Enumerate all temporal walks; positive durations make the search finite."""
    earliest = [math.inf] * vertex_count
    shortest = [math.inf] * vertex_count
    fastest = [math.inf] * vertex_count
    earliest[source] = source_time
    shortest[source] = 0.0
    fastest[source] = 0.0

    outgoing: Dict[int, List[int]] = {v: [] for v in range(vertex_count)}
    for edge_id, u in enumerate(src):
        outgoing[u].append(edge_id)

    def visit(vertex: int, available: int, cost: float, first_departure: int, used: Tuple[int, ...]) -> None:
        for edge_id in outgoing[vertex]:
            if start[edge_id] < available or edge_id in used:
                continue
            edge_arrival = arrival[edge_id]
            edge_cost = cost + weight[edge_id]
            edge_departure = start[edge_id] if not used else first_departure
            earliest[dst[edge_id]] = min(earliest[dst[edge_id]], edge_arrival)
            shortest[dst[edge_id]] = min(shortest[dst[edge_id]], edge_cost)
            fastest[dst[edge_id]] = min(fastest[dst[edge_id]], float(edge_arrival - edge_departure))
            visit(dst[edge_id], edge_arrival, edge_cost, edge_departure, used + (edge_id,))

    visit(source, source_time, 0.0, source_time, ())
    return earliest, shortest, fastest


def run_once(rng: random.Random, iteration: int) -> None:
    vertex_count = rng.randint(2, 7)
    edge_count = rng.randint(1, 16)
    src: List[int] = []
    dst: List[int] = []
    start: List[int] = []
    arrival: List[int] = []
    weight: List[float] = []
    for _ in range(edge_count):
        u = rng.randrange(vertex_count)
        v = rng.randrange(vertex_count)
        while v == u and vertex_count > 1:
            v = rng.randrange(vertex_count)
        t = rng.randrange(0, 12)
        duration = rng.randrange(1, 5)
        src.append(u)
        dst.append(v)
        start.append(t)
        arrival.append(t + duration)
        weight.append(float(rng.randrange(1, 8)))

    graph = CompactTemporalDAG.from_arrays(
        src, dst, start, arrival, weight, vertex_count=vertex_count, sort_edges=True
    )
    source = 0
    source_time = rng.randrange(0, 4)
    expected_arrival, expected_cost, expected_fastest = brute_force(
        src, dst, start, arrival, weight, vertex_count, source, source_time
    )
    got_arrival = mst_a(graph, source, source_time=source_time).arrival
    got_cost = shortest_path(graph, source, source_time=source_time).values
    got_fastest = fastest_path(graph, source, departure_time=source_time).values

    for vertex in range(vertex_count):
        if expected_arrival[vertex] == math.inf:
            assert got_arrival[vertex] == graph.time_max, (iteration, vertex)
        else:
            assert int(got_arrival[vertex]) == int(expected_arrival[vertex]), (iteration, vertex, expected_arrival, got_arrival)
        if math.isinf(expected_cost[vertex]):
            assert math.isinf(float(got_cost[vertex])), (iteration, vertex)
        else:
            assert math.isclose(float(got_cost[vertex]), expected_cost[vertex]), (iteration, vertex, expected_cost, got_cost)
        if math.isinf(expected_fastest[vertex]):
            assert math.isinf(float(got_fastest[vertex])), (iteration, vertex)
        else:
            assert math.isclose(float(got_fastest[vertex]), expected_fastest[vertex]), (iteration, vertex, expected_fastest, got_fastest)

    cost, tree, query = mst_w_upper_bound(graph, source, source_time=source_time)
    reachable = sum(value < math.inf for value in expected_arrival)
    assert len(query.reachable_vertices()) == reachable
    if reachable == 1:
        assert cost == 0.0 and tree == []
    else:
        assert math.isfinite(cost) or reachable < vertex_count


def main() -> None:
    rng = random.Random(20260904)
    for iteration in range(300):
        run_once(rng, iteration)
    print("random differential tests: PASS (300 graphs)")


if __name__ == "__main__":
    sys.exit(main())
