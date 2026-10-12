#!/usr/bin/env python3
"""Reproducible synthetic MST_w quality and representation ablations.

Every measured repetition runs in a fresh process. Timings exclude synthetic
input generation, imports, output validation, and process startup; pruning is
included in total_s. These are small-instance baseline experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from temporal_mst_exact import exact_mst_w, incoming_weight_lower_bound
from temporal_mst_pruning import prune_dominated_events
from temporal_mst_prototype import (
    TemporalEdge,
    TemporalGraph,
    TimeStateDAG,
    shortest_path_tree_upper_bound,
    temporal_reachable,
    validate_temporal_tree,
)


METHODS = ("path_union", "explicit", "channel_explicit", "implicit", "pruned_implicit")


def make_instance(family: str, size: int, seed: int) -> TemporalGraph:
    if family == "sharing_gap":
        vertices = ["r", "hub"] + [f"leaf{position}" for position in range(size)]
        edges = [
            TemporalEdge("early_trunk", "r", "hub", 0, 1, 100),
            TemporalEdge("late_hub", "r", "hub", 10, 11, 1),
        ]
        for vertex in vertices[2:]:
            edges.extend((
                TemporalEdge(f"direct_{vertex}", "r", vertex, 0, 2, 99),
                TemporalEdge(f"shared_{vertex}", "hub", vertex, 1, 2, 1),
            ))
        return TemporalGraph(vertices, edges)
    rng = random.Random(seed)
    vertices = ["r"] + [f"v{position}" for position in range(1, size)]
    edges = [TemporalEdge(f"root_{vertex}", "r", vertex, 0, 1, rng.randrange(10, 40)) for vertex in vertices[1:]]
    for position in range(3 * size):
        tail, head = rng.sample(vertices, 2)
        start = rng.randrange(1, 15)
        edges.append(TemporalEdge(
            f"event_{position}", tail, head, start, start + rng.randrange(1, 5), rng.randrange(1, 30),
        ))
    if family == "redundant":
        original = list(edges)
        for edge in original:
            for extra in range(1, 5):
                edges.append(TemporalEdge(
                    f"{edge.eid}_dominated_{extra}", edge.u, edge.v,
                    edge.start - extra, edge.arrival + extra, edge.weight + extra,
                ))
    elif family != "weighted_random":
        raise ValueError(f"unknown instance family {family}")
    return TemporalGraph(vertices, edges)


def worker(family: str, size: int, seed: int, method: str) -> dict:
    graph = make_instance(family, size, seed)
    started = time.perf_counter()
    work_graph = graph
    prune_s = 0.0
    removed = 0
    if method == "pruned_implicit":
        work_graph, witnesses = prune_dominated_events(graph)
        removed = len(witnesses)
        prune_s = time.perf_counter() - started
    if method == "path_union":
        dag = TimeStateDAG(work_graph, "r")
        cost, edge_ids = shortest_path_tree_upper_bound(dag)
        extra = {"state_count": len(dag.adj), "dp_cells": 0}
    else:
        result = exact_mst_w(
            work_graph, "r", backend="implicit" if method == "pruned_implicit" else method,
        )
        cost, edge_ids = result.cost, result.edge_ids
        extra = {
            key: getattr(result, key) for key in (
                "state_count", "paid_arc_count", "explicit_zero_arc_count",
                "implicit_wait_count", "dp_cells",
            )
        }
    total_s = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    required = temporal_reachable(graph, "r", graph.edges)
    selected = [edge for edge in graph.edges if edge.eid in set(edge_ids)]
    if not validate_temporal_tree(graph, "r", selected, required):
        raise AssertionError("invalid output temporal tree")
    if abs(sum(edge.weight for edge in selected) - cost) > 1e-9:
        raise AssertionError("tree cost mismatch")
    return {
        "family": family, "size": size, "seed": seed, "method": method,
        "vertices": len(graph.vertices), "reachable": len(required),
        "input_edges": len(graph.edges), "retained_edges": len(work_graph.edges),
        "removed_edges": removed, "cost": cost, "edge_ids": sorted(edge_ids),
        "incoming_lower_bound": incoming_weight_lower_bound(graph, "r"),
        "prune_s": prune_s, "solve_s": total_s - prune_s, "total_s": total_s,
        "peak_rss_mib": peak / (1024 * 1024 if sys.platform == "darwin" else 1024),
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--family", default="sharing_gap")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--method", choices=METHODS, default="implicit")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, default=Path("benchmark_mst_research.json"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.family, args.size, args.seed, args.method), allow_nan=False))
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    scenarios = [("sharing_gap", size, args.seed) for size in (2, 4, 6, 8)]
    scenarios.extend((family, 8, seed) for family in ("weighted_random", "redundant") for seed in range(args.seed, args.seed + 3))
    records = []
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    for family, size, seed in scenarios:
        for method in METHODS:
            for repeat in range(args.repeats):
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker",
                    "--family", family, "--size", str(size), "--seed", str(seed), "--method", method,
                ]
                process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, env=environment)
                if process.returncode:
                    raise RuntimeError(process.stderr)
                record = json.loads(process.stdout)
                record["repeat"] = repeat
                records.append(record)
        print(f"completed {family} size={size} seed={seed}", file=sys.stderr, flush=True)
    summaries = []
    for family, size, seed in scenarios:
        scenario = [row for row in records if (row["family"], row["size"], row["seed"]) == (family, size, seed)]
        exact_costs = {row["cost"] for row in scenario if row["method"] != "path_union"}
        if len(exact_costs) != 1:
            raise AssertionError("exact backends disagree")
        optimum = exact_costs.pop()
        for method in METHODS:
            rows = [row for row in scenario if row["method"] == method]
            if any(row["cost"] < optimum or row["incoming_lower_bound"] > optimum for row in rows):
                raise AssertionError("inconsistent lower/upper bound")
            summary = {key: value for key, value in rows[0].items() if key not in {"repeat", "edge_ids"}}
            for key in ("prune_s", "solve_s", "total_s", "peak_rss_mib"):
                summary[key] = statistics.median(row[key] for row in rows)
            summary["optimum"] = optimum
            summary["cost_ratio"] = summary["cost"] / optimum if optimum else 1.0
            summaries.append(summary)
    sources = [Path(__file__), Path(__file__).with_name("temporal_mst_exact.py"), Path(__file__).with_name("temporal_mst_pruning.py"), Path(__file__).with_name("temporal_mst_prototype.py")]
    payload = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"), "python": sys.version, "platform": platform.platform(),
            "repeats": args.repeats, "seed": args.seed, "isolated_processes": True,
            "timing_scope": "index construction + solve + pruning; excludes input generation, imports, validation, process launch",
            "data_scope": "synthetic correctness and ablation only; not a real-data scalability claim",
            "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        },
        "summaries": summaries, "runs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"wrote {len(records)} validated runs to {args.output}")


if __name__ == "__main__":
    main()
