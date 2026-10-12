#!/usr/bin/env python3
"""Benchmark exact and shared-cost temporal MST methods on weighted data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from temporal_mst_exact import exact_mst_w, incoming_weight_lower_bound
from temporal_mst_ilp import ikuta_ilp
from temporal_mst_lagrangian import shared_availability_lagrangian
from temporal_mst_lns import shared_cost_lns
from temporal_mst_prototype import (
    TemporalEdge,
    TemporalGraph,
    earliest_arrival_tree,
    temporal_reachable,
    validate_temporal_tree,
)
from temporal_mst_shared import shared_prefix_optimize, shortest_path_union_tree


def choose_root(graph: TemporalGraph, candidate_count: int = 32) -> Tuple[str, int]:
    outdegree: Dict[str, int] = {vertex: 0 for vertex in graph.vertices}
    first_departure: Dict[str, float] = {vertex: math.inf for vertex in graph.vertices}
    for edge in graph.edges:
        outdegree[edge.u] += 1
        first_departure[edge.u] = min(first_departure[edge.u], edge.start)
    candidates = sorted(
        graph.vertices,
        key=lambda vertex: (-outdegree[vertex], first_departure[vertex], vertex),
    )[:candidate_count]
    best_root = candidates[0]
    best_reach = -1
    for candidate in candidates:
        reach = len(temporal_reachable(graph, candidate, graph.edges))
        if reach > best_reach:
            best_root = candidate
            best_reach = reach
    return best_root, best_reach


def induced_temporal_sample(
    graph: TemporalGraph,
    root: str,
    vertex_limit: int,
) -> TemporalGraph:
    arrival, parent = earliest_arrival_tree(graph, root)
    ordered = sorted(
        (vertex for vertex in graph.vertices if arrival[vertex] < math.inf),
        key=lambda vertex: (arrival[vertex], vertex),
    )
    selected = {root}
    for vertex in ordered:
        if len(selected) >= vertex_limit:
            break
        cursor = vertex
        chain = []
        while cursor != root and cursor not in selected:
            edge = parent[cursor]
            chain.append(cursor)
            cursor = edge.u
        if len(selected) + len(chain) <= vertex_limit:
            selected.update(chain)
    edges = [edge for edge in graph.edges if edge.u in selected and edge.v in selected]
    return TemporalGraph(
        [vertex for vertex in graph.vertices if vertex in selected],
        edges,
    )


def _validated_cost(graph: TemporalGraph, root: str, edge_ids: Sequence[str]) -> float:
    selected_set = set(edge_ids)
    selected = [edge for edge in graph.edges if edge.eid in selected_set]
    required = temporal_reachable(graph, root, graph.edges)
    if not validate_temporal_tree(graph, root, selected, required):
        raise AssertionError("benchmark method returned an invalid temporal tree")
    return math.fsum(edge.weight for edge in selected)


def run_method(graph: TemporalGraph, root: str, method: str, time_limit: float) -> dict:
    started = time.perf_counter()
    if method == "path_union":
        cost, tree, _ = shortest_path_union_tree(graph, root)
        result = {"cost": cost, "edge_ids": sorted(edge.eid for edge in tree)}
    elif method == "shared_prefix":
        result = asdict(
            shared_prefix_optimize(
                graph,
                root,
                max_rounds=2,
                max_hub_states=32,
                max_states_per_vertex=4,
                evaluated_prefixes=8,
            )
        )
    elif method == "shared_lns":
        result = asdict(
            shared_cost_lns(
                graph,
                root,
                max_rounds=2,
                max_hub_states=24,
                max_states_per_vertex=4,
                max_neighborhood=9,
                max_dp_cells=4_000_000,
            )
        )
    elif method == "lagrangian_pool":
        result = asdict(
            shared_availability_lagrangian(
                graph,
                root,
                iterations=200,
                primal_interval=20,
                step_scale=1.8,
                pool_width=4,
                pool_interval=10,
                pool_time_limit=time_limit,
            )
        )
    elif method == "subset_dp":
        result = asdict(exact_mst_w(graph, root, backend="implicit"))
    elif method == "ikuta_ilp":
        result = asdict(ikuta_ilp(graph, root, time_limit=time_limit))
    else:
        raise ValueError(f"unknown method: {method}")
    elapsed = time.perf_counter() - started
    verified_cost = _validated_cost(graph, root, result["edge_ids"])
    if not math.isclose(verified_cost, result["cost"], rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError("reported and reconstructed costs disagree")
    result.update(
        method=method,
        wall_time_s=elapsed,
        vertices=len(graph.vertices),
        edges=len(graph.edges),
        reachable=len(temporal_reachable(graph, root, graph.edges)),
        incoming_lower_bound=incoming_weight_lower_bound(graph, root),
    )
    return result


def write_results(output: Path, repeats: int, records: Sequence[dict]) -> None:
    source_files = [
        Path(__file__),
        Path(__file__).with_name("temporal_mst_ilp.py"),
        Path(__file__).with_name("temporal_mst_shared.py"),
        Path(__file__).with_name("temporal_mst_lagrangian.py"),
        Path(__file__).with_name("temporal_mst_lns.py"),
        Path(__file__).with_name("prepare_weighted_datasets.py"),
        Path(__file__).with_name("prepare_citibike_weighted.py"),
    ]
    payload = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "python": sys.version,
            "platform": platform.platform(),
            "repeats": repeats,
            "sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_files
            },
        },
        "records": list(records),
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    csv_path = output.with_suffix(".csv")
    scalar_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if not isinstance(value, (list, dict))
        }
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in scalar_keys})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--root")
    parser.add_argument("--exact-vertices", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--full-ilp", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("weighted_mst_results.json"))
    args = parser.parse_args()
    records: List[dict] = []
    for input_path in args.inputs:
        graph = TemporalGraph.from_json(str(input_path))
        root, full_reachable = (
            (args.root, len(temporal_reachable(graph, args.root, graph.edges)))
            if args.root is not None
            else choose_root(graph)
        )
        exact_graph = induced_temporal_sample(graph, root, args.exact_vertices)
        full_methods = [
            "path_union",
            "shared_prefix",
            "shared_lns",
            "lagrangian_pool",
        ]
        if args.full_ilp:
            full_methods.append("ikuta_ilp")
        for scope, work_graph, methods in (
            ("exact_sample", exact_graph, ("path_union", "shared_prefix", "shared_lns", "subset_dp", "ikuta_ilp")),
            ("full", graph, tuple(full_methods)),
        ):
            for method in methods:
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    try:
                        record = run_method(work_graph, root, method, args.time_limit)
                    except TimeoutError as exc:
                        if method != "ikuta_ilp":
                            raise
                        record = {
                            "method": method,
                            "status": "timeout-no-incumbent",
                            "error": f"{type(exc).__name__}: {exc}",
                            "wall_time_s": time.perf_counter() - started,
                            "vertices": len(work_graph.vertices),
                            "edges": len(work_graph.edges),
                            "reachable": len(
                                temporal_reachable(work_graph, root, work_graph.edges)
                            ),
                            "incoming_lower_bound": incoming_weight_lower_bound(
                                work_graph, root
                            ),
                        }
                    record.update(
                        dataset=input_path.stem,
                        input=str(input_path),
                        root=root,
                        scope=scope,
                        repeat=repeat,
                        full_reachable=full_reachable,
                    )
                    records.append(record)
                    if "cost" in record:
                        message = f"cost={record['cost']:.6g}"
                    else:
                        message = record["status"]
                    print(
                        f"{input_path.stem} {scope} {method}: "
                        f"{message} time={record['wall_time_s']:.3f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    write_results(args.output, args.repeats, records)
    csv_path = args.output.with_suffix(".csv")
    print(f"wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
