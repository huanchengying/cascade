#!/usr/bin/env python3
"""Fresh-process audit of the frozen authors' fast-level-two implementation."""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def portable_source_key(path):
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def worker(args):
    from temporal_mst_prototype import TemporalGraph
    from temporal_mst_speedsteiner import speedsteiner_tree

    started = time.monotonic()
    graph = TemporalGraph.from_json(str(args.input))
    record = {"method": "speedsteiner", "root": args.root, "vertices": len(graph.vertices), "edges": len(graph.edges), "budget_s": args.budget}
    try:
        result = speedsteiner_tree(graph, args.root, time_limit=max(1e-6, args.budget - (time.monotonic() - started)))
        ids = result.pop("edge_ids")
        record.update(result, validated=True, status="feasible", solution_sha256=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest())
    except TimeoutError as exc:
        record.update(validated=False, status="timeout-no-incumbent", error=str(exc))
    except ValueError as exc:
        if "memory guard" not in str(exc):
            raise
        record.update(validated=False, status="size-guard", error=str(exc))
    elapsed = time.monotonic() - started
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    record.update(wall_time_s=elapsed, deadline_exceeded=elapsed > args.budget, peak_rss_mib=peak / (1024 * 1024 if sys.platform == "darwin" else 1024))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--root")
    parser.add_argument("--budget", type=float, default=30)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("speedsteiner_results.json"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    config = json.loads(args.config.read_text())
    config["methods"] = ["speedsteiner"]
    sources = [Path(__file__), Path("temporal_mst_speedsteiner.py"), Path("speedsteiner_adapter.cpp"), Path("temporal_mst_prototype.py"), Path("work/bin/speedsteiner_adapter")]
    metadata = {"date": time.strftime("%Y-%m-%d"), "config": config, "fresh_process": True, "timing_scope": "input + temporal transformation + author binary launch/solve + original-graph validation; Python imports excluded", "sha256": {portable_source_key(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, "input_sha256": {entry["input"]: hashlib.sha256(Path(entry["input"]).read_bytes()).hexdigest() for entry in config["datasets"]}, "upstream_repository": "https://github.com/Guangyi-Zhang/directed-steiner-tree-code", "upstream_commit": "4191b7cf54b9310f065540dd80fc587aeefd3033", "method": "fast_level2", "cost_precision": "upstream float; reported original double cost", "rss_scope": "maximum of Python and child peaks; not their concurrent sum"}
    records = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for entry in config["datasets"]:
        for root in entry["roots"]:
            for repeat in range(config.get("repeats", 3)):
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--input", entry["input"], "--root", root, "--budget", str(config["budget_s"])]
                process = subprocess.run(command, capture_output=True, text=True, env=env, timeout=2 * config["budget_s"] + 20)
                if process.returncode:
                    raise RuntimeError(process.stderr[-4000:])
                record = json.loads(process.stdout)
                record.update(dataset=entry["name"], repeat=repeat)
                records.append(record)
                args.output.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2, allow_nan=False) + "\n")
                print(f"{entry['name']} root={root} repeat={repeat} {record['status']} cost={record.get('cost')} time={record['wall_time_s']:.3f}", flush=True)


if __name__ == "__main__":
    main()
