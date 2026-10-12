#!/usr/bin/env python3
"""Fresh-process matched-incumbent audit for cascade-star improvements."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


METHODS = ("star", "cascade")


def worker(args):
    from temporal_mst_cascade_star import cascade_activation_star
    from temporal_mst_prototype import TemporalGraph, temporal_reachable, validate_temporal_tree
    from temporal_mst_star import shared_activation_star

    started = time.monotonic()
    graph = TemporalGraph.from_json(str(args.input))
    required = temporal_reachable(graph, args.root, graph.edges)
    if args.method == "star":
        result = asdict(
            shared_activation_star(
                graph,
                args.root,
                max_rounds=args.conservative_rounds,
            )
        )
        result["conservative_star_cost"] = result["cost"]
        result["cascade_moves"] = 0
        result["relaxed_deadline_vertices"] = 0
    else:
        result = asdict(
            cascade_activation_star(
                graph,
                args.root,
                conservative_rounds=args.conservative_rounds,
                cascade_rounds=args.cascade_rounds,
                max_hubs=args.max_hubs,
                max_parent_evaluations=args.max_parent_evaluations,
            )
        )
        result["cascade_moves"] = result["accepted_moves"]
    selected_ids = set(result.pop("edge_ids"))
    result.pop("reachable", None)
    selected = [edge for edge in graph.edges if edge.eid in selected_ids]
    if not validate_temporal_tree(graph, args.root, selected, required):
        raise RuntimeError("worker returned an invalid original-graph temporal tree")
    result.update(
        root=args.root,
        vertices=len(graph.vertices),
        edges=len(graph.edges),
        reachable_count=len(required),
        wall_time_s=time.monotonic() - started,
        method=args.method,
        conservative_rounds_requested=args.conservative_rounds,
        cascade_rounds_requested=args.cascade_rounds,
        max_hubs=args.max_hubs,
        max_parent_evaluations=args.max_parent_evaluations,
        validated=True,
        solution_sha256=hashlib.sha256(
            "\n".join(sorted(selected_ids)).encode()
        ).hexdigest(),
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_mib"] = peak / (
        1024 * 1024 if sys.platform == "darwin" else 1024
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--root")
    parser.add_argument("--method", choices=METHODS, default="cascade")
    parser.add_argument("--conservative-rounds", type=int, default=32)
    parser.add_argument("--cascade-rounds", type=int, default=4)
    parser.add_argument("--max-hubs", type=int, default=128)
    parser.add_argument("--max-parent-evaluations", type=int, default=512)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("cascade_star_results.json")
    )
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
                "temporal_mst_cascade_star.py",
                "temporal_mst_star.py",
                "temporal_mst_shared.py",
                "temporal_mst_prototype.py",
            )
        ],
    ]
    metadata = {
        "date": time.strftime("%Y-%m-%d"),
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "input_sha256": {
            entry["input"]: hashlib.sha256(Path(entry["input"]).read_bytes()).hexdigest()
            for entry in config["datasets"]
        },
        "fresh_process": True,
        "threads": 1,
        "timing_scope": "input + initial tree + local search + validation; imports excluded",
        "paired_unit": "dataset-root; repeats are timing repeats, not independent instances",
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
            for method in config.get("methods", METHODS):
                for repeat in range(config.get("repeats", 1)):
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--input",
                        entry["input"],
                        "--root",
                        root,
                        "--method",
                        method,
                        "--conservative-rounds",
                        str(config.get("conservative_rounds", 32)),
                        "--cascade-rounds",
                        str(config.get("cascade_rounds", 4)),
                        "--max-hubs",
                        str(config.get("max_hubs", 128)),
                        "--max-parent-evaluations",
                        str(config.get("max_parent_evaluations", 512)),
                    ]
                    deadline = config.get("hard_timeout_s", 180)
                    try:
                        process = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            env=environment,
                            timeout=deadline,
                        )
                        if process.returncode:
                            raise RuntimeError(process.stderr[-4000:])
                        record = json.loads(process.stdout)
                    except subprocess.TimeoutExpired:
                        record = {
                            "root": root,
                            "method": method,
                            "status": "hard-timeout",
                            "validated": False,
                            "wall_time_s": deadline,
                        }
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
                        f"{entry['name']} root={root} {method} repeat={repeat} "
                        f"cost={record.get('cost')} time={record['wall_time_s']:.3f}",
                        flush=True,
                    )
    print(f"completed {len(records)} fresh-process runs: {args.output}")


if __name__ == "__main__":
    main()
