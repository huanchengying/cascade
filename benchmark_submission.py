#!/usr/bin/env python3
"""Isolated multi-root, equal-budget comparison with honest bound scopes.

Reported wall time includes loading input, constructing models, solving and
verifying a returned physical tree. Process launch/import time is excluded.
Every run is a fresh process with a fixed hash seed and one BLAS thread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


def worker(args):
    from temporal_mst_certificate import build_prefix_model, certified_pool_optimize, solve_prefix_lp
    from temporal_mst_cascade_star import cascade_activation_star
    from temporal_mst_huang import huang_recursive_greedy
    from temporal_mst_ilp import ikuta_ilp
    from temporal_mst_prototype import TemporalGraph, temporal_reachable, validate_temporal_tree
    from temporal_mst_shared import shortest_path_union_tree
    from temporal_mst_star import shared_activation_star

    started = time.perf_counter()
    graph = TemporalGraph.from_json(str(args.input))
    required = temporal_reachable(graph, args.root, graph.edges)
    record = {"method": args.method, "root": args.root, "vertices": len(graph.vertices), "edges": len(graph.edges), "reachable": len(required), "budget_s": args.budget}
    try:
        if args.method == "path_union":
            cost, tree, _ = shortest_path_union_tree(graph, args.root)
            result = {"cost": cost, "edge_ids": [edge.eid for edge in tree]}
        elif args.method == "star":
            result = asdict(
                shared_activation_star(
                    graph,
                    args.root,
                    max_rounds=args.star_rounds,
                )
            )
        elif args.method == "cascade":
            result = asdict(
                cascade_activation_star(
                    graph,
                    args.root,
                    conservative_rounds=args.star_rounds,
                    cascade_rounds=args.cascade_rounds,
                    max_hubs=args.cascade_max_hubs,
                    max_parent_evaluations=args.cascade_max_parent_evaluations,
                )
            )
        elif args.method in {"ikuta", "ikuta_precut"}:
            initial = None
            if args.method == "ikuta_precut":
                model = build_prefix_model(graph, args.root)
                lp = solve_prefix_lp(model, cut_mode="root", max_rounds=3, max_probes=args.probes, time_limit=max(0.01, 0.25 * args.budget))
                initial = [cut.side for cut in lp.cuts]
                record.update(root_lp_bound=lp.lower_bound, root_lp_cuts=len(initial), root_lp_time_s=lp.wall_time_s)
            remaining = args.budget - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("budget exhausted before MILP")
            result = asdict(ikuta_ilp(graph, args.root, time_limit=remaining, initial_cut_sets=initial))
        elif args.method.startswith("closure_"):
            mode = args.method[len("closure_"):]
            remaining = args.budget - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("budget exhausted before closure optimization")
            result = asdict(certified_pool_optimize(graph, args.root, time_limit=remaining, cut_mode=mode, max_lp_rounds=3, max_probes=args.probes))
        elif args.method in {"huang2", "huang3"}:
            result = asdict(huang_recursive_greedy(graph, args.root, level=int(args.method[-1]), time_limit=max(0.01, args.budget - (time.perf_counter() - started))))
        else:
            raise ValueError("unknown method")
        selected_ids = set(result.pop("edge_ids"))
        selected = [edge for edge in graph.edges if edge.eid in selected_ids]
        if not validate_temporal_tree(graph, args.root, selected, required):
            raise RuntimeError("invalid original-graph temporal tree")
        if not math.isclose(math.fsum(edge.weight for edge in selected), result["cost"], rel_tol=1e-9):
            raise RuntimeError("original cost mismatch")
        result.pop("reachable", None)
        record.update(result)
        record.update(status=record.get("status", "feasible"), validated=True, solution_sha256=hashlib.sha256("\n".join(sorted(selected_ids)).encode()).hexdigest())
    except TimeoutError as exc:
        record.update(status="timeout-no-incumbent", error=str(exc), validated=False)
    except ValueError as exc:
        if "metric closure guard" not in str(exc):
            raise
        record.update(status="size-guard", error=str(exc), validated=False)
    record["wall_time_s"] = time.perf_counter() - started
    record["deadline_exceeded"] = record["wall_time_s"] > args.budget
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["peak_rss_mib"] = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--root")
    parser.add_argument("--method")
    parser.add_argument("--budget", type=float, default=20)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--star-rounds", type=int, default=32)
    parser.add_argument("--cascade-rounds", type=int, default=4)
    parser.add_argument("--cascade-max-hubs", type=int, default=10_000)
    parser.add_argument("--cascade-max-parent-evaluations", type=int, default=100_000)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submission_results.json"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    if args.config is None:
        parser.error("--config is required")
    configuration = json.loads(args.config.read_text())
    records = []
    source_files = [Path(__file__), *[Path(__file__).with_name(name) for name in ("temporal_mst_cascade_star.py", "temporal_mst_certificate.py", "temporal_mst_huang.py", "temporal_mst_ilp.py", "temporal_mst_prototype.py", "temporal_mst_shared.py", "temporal_mst_star.py")]]
    metadata = {
        "date": time.strftime("%Y-%m-%d"), "python": sys.version, "platform": platform.platform(),
        "configuration": configuration, "fresh_process_per_run": True,
        "timing_scope": "input loading + preprocessing + solve + physical-tree validation; excludes import/process launch",
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "input_sha256": {entry["input"]: hashlib.sha256(Path(entry["input"]).read_bytes()).hexdigest() for entry in configuration["datasets"]},
    }
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for dataset in configuration["datasets"]:
        for root in dataset["roots"]:
            for method in configuration["methods"]:
                for repeat in range(configuration.get("repeats", 3)):
                    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--input", dataset["input"], "--root", root, "--method", method, "--budget", str(configuration.get("budget_s", args.budget)), "--probes", str(configuration.get("probes", args.probes)), "--star-rounds", str(configuration.get("star_rounds", 32)), "--cascade-rounds", str(configuration.get("cascade_rounds", 4)), "--cascade-max-hubs", str(configuration.get("cascade_max_hubs", 10_000)), "--cascade-max-parent-evaluations", str(configuration.get("cascade_max_parent_evaluations", 100_000))]
                    deadline = max(30, 2 * configuration.get("budget_s", args.budget) + 20)
                    try:
                        process = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=deadline)
                        if process.returncode:
                            raise RuntimeError(process.stderr[-4000:])
                        record = json.loads(process.stdout)
                        if process.stderr:
                            record["stderr"] = process.stderr[-2000:]
                    except subprocess.TimeoutExpired:
                        record = {"root": root, "method": method, "status": "hard-timeout", "validated": False, "wall_time_s": deadline}
                    record.update(dataset=dataset["name"], repeat=repeat)
                    records.append(record)
                    args.output.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2, allow_nan=False) + "\n")
                    print(f"{dataset['name']} root={root} {method} repeat={repeat}: {record['status']} cost={record.get('cost')} time={record['wall_time_s']:.3f}", flush=True)
    summaries = []
    for dataset in configuration["datasets"]:
        for root in dataset["roots"]:
            for method in configuration["methods"]:
                rows = [row for row in records if (row["dataset"], row["root"], row["method"]) == (dataset["name"], root, method)]
                feasible = [row for row in rows if row.get("validated")]
                summary = {"dataset": dataset["name"], "root": root, "method": method, "runs": len(rows), "feasible_runs": len(feasible), "optimal_runs": sum(row.get("certified_optimal", row.get("optimal", False)) for row in feasible), "median_wall_s": statistics.median(row["wall_time_s"] for row in rows)}
                for key in ("cost", "lower_bound", "peak_rss_mib", "safe_edges", "original_edges", "pool_edges", "root_cuts", "deadline_cuts"):
                    values = [row[key] for row in feasible if key in row]
                    if values:
                        summary["median_" + key] = statistics.median(values)
                summaries.append(summary)
    args.output.write_text(json.dumps({"metadata": metadata, "records": records, "summaries": summaries}, indent=2, allow_nan=False) + "\n")
    print(f"completed {len(records)} fresh-process runs: {args.output}")


if __name__ == "__main__":
    main()
