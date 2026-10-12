#!/usr/bin/env python3
"""Fresh-process matched-budget audit for shared-star warm starts and pricing."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


METHODS = (
    "star",
    "cascade",
    "star_warm",
    "star_warm_root",
    "cascade_warm",
    "cascade_warm_root",
    "star_closure_weight",
    "star_closure_none",
    "star_closure_root",
    "star_closure_deadline",
    "cascade_closure_weight",
    "cascade_closure_none",
    "cascade_closure_root",
    "cascade_closure_deadline",
)


def worker(args):
    from temporal_mst_prototype import TemporalGraph, temporal_reachable, validate_temporal_tree
    from temporal_mst_cascade_star import cascade_activation_star
    from temporal_mst_star import shared_activation_star
    from temporal_mst_star_priced import star_priced_warm_optimize, star_warm_full

    started = time.monotonic()
    graph = TemporalGraph.from_json(str(args.input))
    required = temporal_reachable(graph, args.root, graph.edges)
    remaining = max(1e-6, args.budget - (time.monotonic() - started))
    initial_mode = "cascade" if args.method.startswith("cascade") else "star"
    if args.method in {"star", "cascade"}:
        if initial_mode == "star":
            initial = shared_activation_star(
                graph, args.root, max_rounds=args.star_rounds
            )
        else:
            initial = cascade_activation_star(
                graph,
                args.root,
                conservative_rounds=args.star_rounds,
                cascade_rounds=args.cascade_rounds,
                max_hubs=args.cascade_max_hubs,
                max_parent_evaluations=args.cascade_max_parent_evaluations,
            )
        result = {
            **asdict(initial),
            "initial_cost": initial.cost,
            "lower_bound": 0.0,
            "relative_gap": 1.0,
            "certified_optimal": False,
            "status": "feasible",
        }
    elif args.method in {
        "star_warm",
        "star_warm_root",
        "cascade_warm",
        "cascade_warm_root",
    }:
        result = star_warm_full(
            graph,
            args.root,
            time_limit=remaining,
            star_rounds=args.star_rounds,
            initial_mode=initial_mode,
            cascade_rounds=args.cascade_rounds,
            cascade_max_hubs=args.cascade_max_hubs,
            cascade_max_parent_evaluations=args.cascade_max_parent_evaluations,
            cut_mode="root" if args.method.endswith("_root") else "none",
            max_probes=args.probes,
        )
    else:
        mode = args.method[len(f"{initial_mode}_closure_"):]
        result = star_priced_warm_optimize(
            graph,
            args.root,
            time_limit=remaining,
            cut_mode=mode,
            max_probes=args.probes,
            star_rounds=args.star_rounds,
            initial_mode=initial_mode,
            cascade_rounds=args.cascade_rounds,
            cascade_max_hubs=args.cascade_max_hubs,
            cascade_max_parent_evaluations=args.cascade_max_parent_evaluations,
        )
    selected_ids = set(result.pop("edge_ids"))
    result.pop("reachable", None)
    selected = [edge for edge in graph.edges if edge.eid in selected_ids]
    if not validate_temporal_tree(graph, args.root, selected, required):
        raise RuntimeError("worker returned an invalid original-graph temporal tree")
    elapsed = time.monotonic() - started
    result.update(
        root=args.root,
        vertices=len(graph.vertices),
        edges=len(graph.edges),
        reachable_count=len(required),
        wall_time_s=elapsed,
        method=args.method,
        budget_s=args.budget,
        star_rounds_requested=args.star_rounds,
        deadline_exceeded=elapsed > args.budget,
        validated=True,
        solution_sha256=hashlib.sha256("\n".join(sorted(selected_ids)).encode()).hexdigest(),
    )
    result.setdefault("status", "feasible")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_mib"] = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    return result


def asdict(value):
    from dataclasses import asdict as convert

    return convert(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--root")
    parser.add_argument("--method", choices=METHODS, default="star_warm")
    parser.add_argument("--probes", type=int, default=6)
    parser.add_argument("--star-rounds", type=int, default=4)
    parser.add_argument("--cascade-rounds", type=int, default=2)
    parser.add_argument("--cascade-max-hubs", type=int, default=64)
    parser.add_argument("--cascade-max-parent-evaluations", type=int, default=128)
    parser.add_argument("--budget", type=float, default=30)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("star_warm_results.json"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    if args.config is None:
        parser.error("--config is required")
    config = json.loads(args.config.read_text())
    sources = [
        Path(__file__),
        *[
            Path(__file__).with_name(name)
            for name in (
                "temporal_mst_star.py",
                "temporal_mst_cascade_star.py",
                "temporal_mst_star_priced.py",
                "temporal_mst_highs.py",
                "temporal_mst_certificate.py",
                "temporal_mst_ilp.py",
                "temporal_mst_shared.py",
                "temporal_mst_prototype.py",
            )
        ],
    ]
    import highspy
    import numpy
    import scipy

    metadata = {
        "date": time.strftime("%Y-%m-%d"),
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "highspy": highspy.Highs().version(),
            "scipy": scipy.__version__,
            "numpy": numpy.__version__,
        },
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "input_sha256": {
            entry["input"]: hashlib.sha256(Path(entry["input"]).read_bytes()).hexdigest()
            for entry in config["datasets"]
        },
        "fresh_process": True,
        "threads": 1,
        "timing_scope": "input + shared-star incumbent + optional LP + native masters + validation; imports excluded",
    }
    records = []
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for entry in config["datasets"]:
        for root in entry["roots"]:
            for budget in config.get("budgets_s", [config.get("budget_s", 30)]):
                for method in config.get("methods", METHODS):
                    for repeat in range(config.get("repeats", 3)):
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--input",
                            entry["input"],
                            "--root",
                            root,
                            "--budget",
                            str(budget),
                            "--method",
                            method,
                            "--probes",
                            str(config.get("probes", 6)),
                            "--star-rounds",
                            str(config.get("star_rounds", 4)),
                            "--cascade-rounds",
                            str(config.get("cascade_rounds", 2)),
                            "--cascade-max-hubs",
                            str(config.get("cascade_max_hubs", 64)),
                            "--cascade-max-parent-evaluations",
                            str(config.get("cascade_max_parent_evaluations", 128)),
                        ]
                        deadline = 2 * budget + 30
                        failures = []
                        record = None
                        attempts = 1 + config.get("worker_retries", 0)
                        for attempt in range(attempts):
                            attempt_started = time.monotonic()
                            try:
                                process = subprocess.run(
                                    command,
                                    capture_output=True,
                                    text=True,
                                    env=environment,
                                    timeout=deadline,
                                )
                            except subprocess.TimeoutExpired:
                                record = {
                                    "root": root,
                                    "method": method,
                                    "status": "hard-timeout",
                                    "validated": False,
                                    "wall_time_s": deadline,
                                    "budget_s": budget,
                                }
                                break
                            if process.returncode == 0:
                                try:
                                    record = json.loads(process.stdout)
                                    break
                                except json.JSONDecodeError as exc:
                                    error = f"invalid worker JSON: {exc}"
                            else:
                                error = process.stderr[-4000:] or "worker exited without stderr"
                            failures.append(
                                {
                                    "attempt": attempt + 1,
                                    "returncode": process.returncode,
                                    "wall_time_s": time.monotonic() - attempt_started,
                                    "error": error,
                                }
                            )
                            if attempt + 1 < attempts:
                                time.sleep(config.get("retry_delay_s", 0))
                        if record is None:
                            record = {
                                "root": root,
                                "method": method,
                                "status": "worker-error",
                                "validated": False,
                                "wall_time_s": failures[-1]["wall_time_s"],
                                "budget_s": budget,
                            }
                        record["worker_attempts"] = len(failures) + int(
                            record["status"] != "worker-error"
                        )
                        if failures:
                            record["worker_failures"] = failures
                        record.update(dataset=entry["name"], repeat=repeat)
                        records.append(record)
                        args.output.write_text(
                            json.dumps(
                                {"metadata": metadata, "records": records},
                                indent=2,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        print(
                            f"{entry['name']} root={root} budget={budget} "
                            f"{method} repeat={repeat} cost={record.get('cost')} "
                            f"gap={record.get('relative_gap', 0):.4f} "
                            f"time={record['wall_time_s']:.3f}",
                            flush=True,
                        )
    print(f"completed {len(records)} fresh-process runs: {args.output}")


if __name__ == "__main__":
    main()
