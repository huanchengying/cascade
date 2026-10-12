#!/usr/bin/env python3
"""Compare shortest-path execution across temporal-DAG representations."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import platform
import resource
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from benchmark_sx_stackoverflow import load_prefix
from temporal_dag_framework import CompactTemporalDAG, shortest_path
from temporal_mst_prototype import (
    LazyTeGraphIndex,
    TeGraphDAG,
    TimeStateDAG,
)


DEFAULT_INPUT = Path(__file__).with_name("sx-stackoverflow")
METHODS = (
    "Huang_style_state_DAG_shortest",
    "TeGraph_style_in_out_DAG_shortest",
    "Proposed_lazy_hyper_DAG_shortest",
    "Compact_DAG_shortest",
)


def explicit_sssp(dag: Any) -> Dict[str, Any]:
    """Topological shortest paths from the DAG source to all terminals."""
    distance: Dict[str, float] = {dag.source: 0.0}
    for node in dag.topological_order():
        base = distance.get(node, math.inf)
        if not math.isfinite(base):
            continue
        for target, edge_cost, _ in dag.adj.get(node, []):
            candidate = base + edge_cost
            if candidate < distance.get(target, math.inf):
                distance[target] = candidate
    values = [distance.get(terminal, math.inf) for terminal in dag.terminals.values()]
    finite = [value for value in values if math.isfinite(value)]
    return {
        "reachable_vertices": len(finite) + 1,
        "max_shortest_cost": max(finite) if finite else None,
    }


def lazy_sssp(index: LazyTeGraphIndex) -> Dict[str, Any]:
    """Object-based lazy temporal SSSP with implicit waiting transitions."""
    edges = [edge for edge in index.ordered_edges if edge.v != index.root]
    monotone = all(left.arrival <= right.arrival for left, right in zip(edges, edges[1:]))
    pending: Any = deque() if monotone else []
    best_cost: Dict[str, float] = {index.root: 0.0}
    sequence = 0
    edge_pos = 0

    while edge_pos < len(edges) or pending:
        next_start = edges[edge_pos].start if edge_pos < len(edges) else math.inf
        while pending and pending[0][0] <= next_start:
            if monotone:
                arrival, _, vertex, cost = pending.popleft()
            else:
                arrival, _, vertex, cost = heapq.heappop(pending)
            if cost < best_cost.get(vertex, math.inf):
                best_cost[vertex] = cost
        if edge_pos == len(edges):
            break
        edge = edges[edge_pos]
        edge_pos += 1
        source_cost = best_cost.get(edge.u, math.inf)
        if not math.isfinite(source_cost):
            continue
        sequence += 1
        item = (edge.arrival, sequence, edge.v, source_cost + edge.weight)
        if monotone:
            pending.append(item)
        else:
            heapq.heappush(pending, item)
    finite = [best_cost[vertex] for vertex in index.reachable if vertex in best_cost]
    return {
        "reachable_vertices": len(finite),
        "max_shortest_cost": max(finite) if finite else None,
    }


def peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input)
    prefix = None if args.prefix == 0 else args.prefix
    started = time.perf_counter()
    if args.method == "Compact_DAG_shortest":
        graph = CompactTemporalDAG.from_event_file(
            input_path,
            duration=args.duration,
            max_edges=prefix,
            root=args.root,
            keep_names=False,
            compress_constants=True,
        )
        build_seconds = time.perf_counter() - started
        solve_started = time.perf_counter()
        result = shortest_path(graph, graph.root_id, source_time=0, return_paths=False)
        solve_seconds = time.perf_counter() - solve_started
        reachable = int(len(result.reachable_vertices()))
        values = result.values
        finite = values[np.isfinite(values)]
        return {
            "method": args.method,
            "status": "ok",
            "edge_count": graph.edge_count,
            "vertex_count": graph.vertex_count,
            "reachable_vertices": reachable,
            "max_shortest_cost": float(np.max(finite)) if len(finite) else None,
            "state_count": graph.state_count,
            "materialized_virtual_arc_count": 0,
            "numeric_memory_mib": graph.numeric_memory_bytes() / (1024 * 1024),
            "build_seconds": build_seconds,
            "solve_seconds": solve_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
            "peak_rss_mib": peak_rss_mib(),
        }

    graph, load_result = load_prefix(input_path, prefix, args.duration)
    if args.root not in graph.vertices:
        raise ValueError(f"root {args.root!r} does not occur in this prefix")
    transform_started = time.perf_counter()
    if args.method == "Huang_style_state_DAG_shortest":
        dag = TimeStateDAG(graph, args.root)
    elif args.method == "TeGraph_style_in_out_DAG_shortest":
        dag = TeGraphDAG(graph, args.root)
    elif args.method == "Proposed_lazy_hyper_DAG_shortest":
        dag = LazyTeGraphIndex(graph, args.root)
    else:
        raise ValueError(args.method)
    transform_seconds = time.perf_counter() - transform_started
    solve_started = time.perf_counter()
    if args.method == "Proposed_lazy_hyper_DAG_shortest":
        shortest_result = lazy_sssp(dag)
        summary = dag.summary()
    else:
        shortest_result = explicit_sssp(dag)
        virtual_arcs = sum(
            1 for arcs in dag.adj.values() for _, _, physical in arcs if physical is None
        )
        summary = {"states": len(dag.states), "materialized_virtual_arcs": virtual_arcs}
    solve_seconds = time.perf_counter() - solve_started
    reachable = int(shortest_result["reachable_vertices"])
    return {
        "method": args.method,
        "status": "ok",
        "edge_count": load_result["edge_count"],
        "vertex_count": load_result["vertex_count"],
        "reachable_vertices": reachable,
        "max_shortest_cost": shortest_result["max_shortest_cost"],
        "state_count": summary["states"],
        "materialized_virtual_arc_count": summary.get("materialized_virtual_arcs", 0),
        "numeric_memory_mib": None,
        "load_seconds": load_result["load_seconds"],
        "transform_seconds": transform_seconds,
        "build_seconds": load_result["load_seconds"] + transform_seconds,
        "solve_seconds": solve_seconds,
        "end_to_end_seconds": load_result["load_seconds"] + transform_seconds + solve_seconds,
        "peak_rss_mib": peak_rss_mib(),
    }


def invoke(args: argparse.Namespace, method: str, prefix: int, repetition: int) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--method",
        method,
        "--input",
        args.input,
        "--prefix",
        str(prefix),
        "--root",
        args.root,
        "--duration",
        str(args.duration),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        return {
            "method": method,
            "status": "worker_failed",
            "error": completed.stderr[-2000:],
            "prefix_edges": prefix,
            "repetition": repetition,
            "worker_wall_seconds": elapsed,
        }
    result = json.loads(completed.stdout)
    result["prefix_edges"] = prefix
    result["repetition"] = repetition
    result["worker_wall_seconds"] = elapsed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--root", default="9")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[100_000])
    parser.add_argument("--explicit-max-edges", type=int, default=100_000)
    parser.add_argument("--lazy-max-edges", type=int, default=1_000_000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", default=str(Path(__file__).with_name("benchmark_shortest_path.json")))
    parser.add_argument("--csv-output", default=str(Path(__file__).with_name("benchmark_shortest_path.csv")))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--prefix", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_worker(args), sort_keys=True))
        return
    records: List[Dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for prefix in sorted(set(args.prefixes)):
            for method in METHODS:
                if method.startswith("Huang_") or method.startswith("TeGraph_"):
                    if prefix > args.explicit_max_edges:
                        continue
                if method.startswith("Proposed_") and prefix > args.lazy_max_edges:
                    continue
                print(f"[{repetition}/{args.repetitions}] {method}, prefix={prefix}", flush=True)
                records.append(invoke(args, method, prefix, repetition))
    payload = {
        "metadata": {
            "input": str(Path(args.input).resolve()),
            "root": args.root,
            "duration": args.duration,
            "objective": "single-source temporal shortest path / shortest-path-tree construction",
            "scope": "local reproductions, not official TeGraph or Huang binaries",
            "platform": platform.platform(),
            "methods": list(METHODS),
            "repetitions": args.repetitions,
        },
        "records": records,
    }
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    fields = sorted({key for record in records for key in record})
    with Path(args.csv_output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
