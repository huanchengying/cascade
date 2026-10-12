#!/usr/bin/env python3
"""Reproducible local benchmark for temporal-MST DAG representations.

The sx-stackoverflow trace contains ``source destination timestamp`` events,
not temporal edges with duration and cost.  This runner uses the explicit
experimental convention ``arrival = timestamp + duration`` and unit edge
weight by default.  Consequently it measures transformation/execution cost;
on this unweighted trace, every rooted temporal tree spanning the same set has
the same total edge count, so it is not an informative MST_w-quality study.

The labels deliberately describe implementations rather than papers:
  * Huang_MST_a_stream: Huang et al.'s earliest-arrival scan semantics;
  * Huang_style_state_DAG: local explicit time-state transformation;
  * TeGraph_style_in_out_DAG: local explicit TeGraph-like transformation;
  * Proposed_lazy_hyper_DAG: local implicit-waiting hyper-DAG index.

This program does NOT benchmark the official TeGraph engine nor Huang et al.'s
full Directed-Steiner approximation.  It is a fair representation benchmark
because all variants run in this process from identical trace prefixes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from temporal_mst_prototype import (
    LazyTeGraphIndex,
    lazy_label_sweep_upper_bound,
    TeGraphDAG,
    TemporalEdge,
    TemporalGraph,
    TimeStateDAG,
    shortest_path_tree_upper_bound,
)


DEFAULT_INPUT = Path(__file__).with_name("sx-stackoverflow")
METHODS = (
    "Huang_MST_a_stream",
    "Huang_MST_a_sorted_scan",
    "Huang_style_state_DAG",
    "TeGraph_style_in_out_DAG",
    "Proposed_lazy_hyper_DAG",
)


def peak_rss_mib() -> float:
    """Normalize ru_maxrss, whose unit differs between macOS and Linux."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def parse_event(line: str, line_number: int) -> Tuple[str, str, float]:
    parts = line.split()
    if len(parts) != 3:
        raise ValueError(f"line {line_number}: expected three fields")
    source, target, raw_time = parts
    timestamp = float(raw_time)
    if not math.isfinite(timestamp):
        raise ValueError(f"line {line_number}: non-finite timestamp")
    return source, target, timestamp


def load_prefix(
    input_path: Path, prefix: Optional[int], duration: float
) -> Tuple[TemporalGraph, Dict[str, Any]]:
    """Load a prefix as unit-weight temporal edges while checking time order."""
    started = time.perf_counter()
    vertices = set()
    edges: List[TemporalEdge] = []
    first_time: Optional[float] = None
    last_time: Optional[float] = None
    sorted_by_start = True
    previous_time = -math.inf
    with input_path.open("r", encoding="ascii") as handle:
        for index, line in enumerate(handle):
            if prefix is not None and index >= prefix:
                break
            source, target, timestamp = parse_event(line, index + 1)
            vertices.add(source)
            vertices.add(target)
            edges.append(
                TemporalEdge(
                    f"e{index}", source, target, timestamp, timestamp + duration, 1.0
                )
            )
            first_time = timestamp if first_time is None else first_time
            last_time = timestamp
            sorted_by_start = sorted_by_start and timestamp >= previous_time
            previous_time = timestamp
    return TemporalGraph(sorted(vertices), edges), {
        "edge_count": len(edges),
        "vertex_count": len(vertices),
        "first_timestamp": first_time,
        "last_timestamp": last_time,
        "trace_sorted_by_start": sorted_by_start,
        "load_seconds": time.perf_counter() - started,
    }


def stream_mst_a(
    input_path: Path,
    prefix: Optional[int],
    root: str,
    duration: float,
    sort_by_time: bool = False,
) -> Dict[str, Any]:
    """Huang MST_a scan, optionally after in-memory timestamp sorting."""
    started = time.perf_counter()
    arrival: Dict[str, float] = {root: -math.inf}
    vertices = {root}
    events: Optional[List[Tuple[str, str, float, int]]] = [] if sort_by_time else None
    selected_edges = 0
    input_edge_count = 0
    first_time: Optional[float] = None
    last_time: Optional[float] = None
    sorted_by_start = True
    previous_time = -math.inf
    sort_seconds = 0.0
    scan_seconds: Optional[float] = None
    with input_path.open("r", encoding="ascii") as handle:
        for index, line in enumerate(handle):
            if prefix is not None and index >= prefix:
                break
            source, target, timestamp = parse_event(line, index + 1)
            input_edge_count += 1
            if events is not None:
                events.append((source, target, timestamp, index))
            elif arrival.get(source, math.inf) <= timestamp:
                candidate = timestamp + duration
                if candidate < arrival.get(target, math.inf):
                    arrival[target] = candidate
                    selected_edges += 1
            vertices.add(source)
            vertices.add(target)
            first_time = timestamp if first_time is None else first_time
            last_time = timestamp
            sorted_by_start = sorted_by_start and timestamp >= previous_time
            previous_time = timestamp
    if events is not None:
        sort_started = time.perf_counter()
        input_edge_count = len(events)
        events.sort(key=lambda event: (event[2], event[3]))
        sort_seconds = time.perf_counter() - sort_started
        scan_events: Iterable[Tuple[str, str, float, int]] = events
    else:
        scan_events = []
    if events is not None:
        scan_started = time.perf_counter()
        for source, target, timestamp, _ in scan_events:
            if arrival.get(source, math.inf) <= timestamp:
                candidate = timestamp + duration
                if candidate < arrival.get(target, math.inf):
                    arrival[target] = candidate
                    selected_edges += 1
        scan_seconds = time.perf_counter() - scan_started
    latest_arrival = max(
        (value for value in arrival.values() if value > -math.inf), default=None
    )
    valid_scan = sort_by_time or sorted_by_start
    final_tree_edges = max(0, len(arrival) - 1) if valid_scan else None
    return {
        "method": "Huang_MST_a_sorted_scan" if sort_by_time else "Huang_MST_a_stream",
        "status": "ok" if valid_scan else "invalid_unsorted_input",
        "order_mode": "in_memory_timestamp_sort" if sort_by_time else "raw_input_order",
        "edge_count": input_edge_count,
        "vertex_count": len(vertices),
        "reachable_vertices": len(arrival),
        "selected_tree_edges": final_tree_edges,
        "relaxation_updates": selected_edges,
        "unit_weight_tree_cost": float(final_tree_edges) if final_tree_edges is not None else None,
        "latest_arrival": latest_arrival if valid_scan else None,
        "first_timestamp": first_time,
        "last_timestamp": last_time,
        "trace_sorted_by_start": sorted_by_start,
        "sort_included": bool(sort_by_time),
        "sort_seconds": sort_seconds,
        "scan_seconds": scan_seconds,
        "end_to_end_seconds": time.perf_counter() - started,
        "peak_rss_mib": peak_rss_mib(),
    }


def explicit_dag_metrics(dag: Any) -> Dict[str, int]:
    real_arcs = 0
    virtual_arcs = 0
    for arcs in dag.adj.values():
        for _, _, physical_edge in arcs:
            if physical_edge is None:
                virtual_arcs += 1
            else:
                real_arcs += 1
    return {
        "state_count": len(dag.states),
        "terminal_count": len(dag.terminals),
        "dag_arc_count": real_arcs + virtual_arcs,
        "real_arc_count": real_arcs,
        "materialized_virtual_arc_count": virtual_arcs,
    }


def run_explicit_method(
    method: str, input_path: Path, prefix: Optional[int], root: str, duration: float,
    solve: bool,
) -> Dict[str, Any]:
    graph, result = load_prefix(input_path, prefix, duration)
    if root not in graph.vertices:
        raise ValueError(f"root {root!r} does not occur in this prefix")
    transform_started = time.perf_counter()
    dag_class = TimeStateDAG if method == "Huang_style_state_DAG" else TeGraphDAG
    dag = dag_class(graph, root)
    transform_seconds = time.perf_counter() - transform_started
    result.update(explicit_dag_metrics(dag))
    result.update(
        {
            "method": method,
            "status": "ok",
            "reachable_vertices": len(dag.reachable),
            "transform_seconds": transform_seconds,
        }
    )
    if solve:
        solve_started = time.perf_counter()
        cost, edge_ids = shortest_path_tree_upper_bound(dag)
        result.update(
            {
                "solution_kind": "single_shortest_path_tree_upper_bound",
                "solution_status": "ok" if cost < math.inf else "no_feasible_tree",
                "unit_weight_tree_cost": cost if cost < math.inf else None,
                "selected_tree_edges": len(edge_ids),
                "solve_seconds": time.perf_counter() - solve_started,
            }
        )
    result["end_to_end_seconds"] = result["load_seconds"] + transform_seconds + result.get("solve_seconds", 0.0)
    result["peak_rss_mib"] = peak_rss_mib()
    return result


def run_lazy_method(
    input_path: Path,
    prefix: Optional[int],
    root: str,
    duration: float,
    solve: bool,
) -> Dict[str, Any]:
    graph, result = load_prefix(input_path, prefix, duration)
    if root not in graph.vertices:
        raise ValueError(f"root {root!r} does not occur in this prefix")
    transform_started = time.perf_counter()
    index = LazyTeGraphIndex(graph, root)
    transform_seconds = time.perf_counter() - transform_started
    summary = index.summary()
    result.update(
        {
            "method": "Proposed_lazy_hyper_DAG",
            "status": "ok",
            "reachable_vertices": len(index.reachable),
            "state_count": summary["states"],
            "real_arc_count": summary["real_arcs"],
            "materialized_virtual_arc_count": summary["materialized_virtual_arcs"],
            "implicit_wait_channels": summary["implicit_wait_channels"],
            "implicit_terminal_channels": summary["implicit_terminal_channels"],
            "tegraph_explicit_virtual_arcs_avoided": summary[
                "tegraph_explicit_virtual_arcs_avoided"
            ],
            "transform_seconds": transform_seconds,
            "end_to_end_seconds": result["load_seconds"] + transform_seconds,
            "peak_rss_mib": peak_rss_mib(),
        }
    )
    if solve:
        solve_started = time.perf_counter()
        cost, edge_ids = lazy_label_sweep_upper_bound(index)
        result.update(
            {
                "solution_kind": "lazy_label_sweep_shortest_path_tree_upper_bound",
                "solution_status": "ok" if cost < math.inf else "no_feasible_tree",
                "unit_weight_tree_cost": cost if cost < math.inf else None,
                "selected_tree_edges": len(edge_ids),
                "solve_seconds": time.perf_counter() - solve_started,
            }
        )
        result["end_to_end_seconds"] += result["solve_seconds"]
    return result


def run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input)
    prefix = None if args.prefix == 0 else args.prefix
    common = {
        "input": str(input_path),
        "prefix_edges": "full" if prefix is None else prefix,
        "root": args.root,
        "duration": args.duration,
        "weight_model": "unit",
        "implementation_scope": "local_reproduction_not_official_baseline",
    }
    if args.method in {"Huang_MST_a_stream", "Huang_MST_a_sorted_scan"}:
        result = stream_mst_a(
            input_path,
            prefix,
            args.root,
            args.duration,
            sort_by_time=args.method == "Huang_MST_a_sorted_scan",
        )
    elif args.method in {"Huang_style_state_DAG", "TeGraph_style_in_out_DAG"}:
        result = run_explicit_method(
            args.method, input_path, prefix, args.root, args.duration, args.solve_explicit
        )
    elif args.method == "Proposed_lazy_hyper_DAG":
        result = run_lazy_method(
            input_path, prefix, args.root, args.duration, args.solve_explicit
        )
    else:
        raise ValueError(f"unknown method: {args.method}")
    return {**common, **result}


def invoke_worker(args: argparse.Namespace, method: str, prefix: int, repetition: int) -> Dict[str, Any]:
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
    if args.solve_explicit:
        command.append("--solve-explicit")
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode == 0:
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            result = {
                "method": method,
                "status": "invalid_worker_json",
                "worker_stdout": completed.stdout[-1000:],
                "worker_stderr": completed.stderr[-1000:],
                "error": str(exc),
            }
    else:
        result = {
            "method": method,
            "status": "worker_failed",
            "worker_exit_code": completed.returncode,
            "worker_stderr": completed.stderr[-2000:],
        }
    result["prefix_edges"] = "full" if prefix == 0 else prefix
    result["repetition"] = repetition
    result["worker_wall_seconds"] = elapsed
    return result


def write_results(records: List[Dict[str, Any]], output_prefix: Path, metadata: Dict[str, Any]) -> None:
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "records": records}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    fields = sorted({field for record in records for field in record})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--root", default="9", help="root vertex; 9 is the first trace source")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[10_000, 50_000, 100_000])
    parser.add_argument("--explicit-max-edges", type=int, default=100_000)
    parser.add_argument("--lazy-max-edges", type=int, default=250_000)
    parser.add_argument("--sorted-max-edges", type=int, default=1_000_000)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--full-stream", action="store_true")
    parser.add_argument("--solve-explicit", action="store_true")
    parser.add_argument("--output-prefix", default=str(Path(__file__).with_name("benchmark_sx_stackoverflow")))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--prefix", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive so the temporal expansion is acyclic")
    if args.worker:
        try:
            print(json.dumps(run_worker(args), sort_keys=True))
        except Exception as exc:  # A parent process records failures as results.
            print(json.dumps({"method": args.method, "status": "worker_exception", "error": repr(exc)}))
            raise
        return
    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"input is not a file: {input_path}")
    prefixes = sorted(set(args.prefixes))
    if any(value <= 0 for value in prefixes):
        parser.error("--prefixes must contain positive edge counts")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    records: List[Dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        # Rotate method order to reduce systematic OS-cache advantage.
        rotation = (repetition - 1) % len(METHODS)
        ordered_methods = METHODS[rotation:] + METHODS[:rotation]
        for prefix in prefixes:
            for method in ordered_methods:
                if method in {"Huang_style_state_DAG", "TeGraph_style_in_out_DAG"} and prefix > args.explicit_max_edges:
                    continue
                if method == "Proposed_lazy_hyper_DAG" and prefix > args.lazy_max_edges:
                    continue
                if method == "Huang_MST_a_sorted_scan" and prefix > args.sorted_max_edges:
                    continue
                print(f"[{repetition}/{args.repetitions}] {method}, prefix={prefix}", flush=True)
                records.append(invoke_worker(args, method, prefix, repetition))
    if args.full_stream:
        print("[full] Huang_MST_a_stream", flush=True)
        records.append(invoke_worker(args, "Huang_MST_a_stream", 0, 1))

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.resolve()),
        "input_bytes": input_path.stat().st_size,
        "root": args.root,
        "duration_convention": "arrival = timestamp + duration",
        "duration": args.duration,
        "weight_convention": "every interaction has unit weight",
        "quality_interpretation": "unit-weight tree cost is edge count; not a discriminative MST_w objective",
        "comparison_scope": "local algorithmic reproductions, not official TeGraph or Huang implementations",
        "platform": platform.platform(),
        "python": sys.version,
        "prefixes": prefixes,
        "repetitions": args.repetitions,
        "explicit_max_edges": args.explicit_max_edges,
        "lazy_max_edges": args.lazy_max_edges,
        "explicit_solver": "single-shortest-path-tree upper bound" if args.solve_explicit else "not run",
        "methods": list(METHODS),
    }
    write_results(records, Path(args.output_prefix), metadata)


if __name__ == "__main__":
    main()
