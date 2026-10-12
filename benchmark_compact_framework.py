#!/usr/bin/env python3
"""Benchmark the compact DAG framework on sx-stackoverflow.

Each application is executed in a fresh process so peak RSS is independent.
The graph is built directly from the event stream with integer vertex IDs and
NumPy/CSR arrays; ``keep_names`` is disabled by default to measure the memory
footprint relevant to out-of-core execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT = Path(__file__).with_name("sx-stackoverflow")
APPLICATIONS = ("reachability", "fastest", "shortest", "mst_a", "mst_w")


def invoke(
    input_path: Path,
    max_edges: int,
    application: str,
    root: str,
    duration: int,
    numeric_direct_ids: bool,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("temporal_dag_framework.py")),
        "--input",
        str(input_path),
        "--root",
        root,
        "--duration",
        str(duration),
        "--max-edges",
        str(max_edges),
        "--application",
        application,
    ]
    if numeric_direct_ids:
        command.append("--numeric-direct-ids")
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - started
    if completed.returncode != 0:
        return {
            "application": application,
            "status": "failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-2000:],
            "wall_seconds": wall,
        }
    try:
        result = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        return {
            "application": application,
            "status": "invalid_output",
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
            "wall_seconds": wall,
        }
    result["application"] = application
    result["status"] = "ok"
    result["worker_wall_seconds"] = wall
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--prefixes", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--root", default="9")
    parser.add_argument("--duration", type=int, default=1)
    parser.add_argument("--output", default=str(Path(__file__).with_name("benchmark_compact_framework.json")))
    parser.add_argument("--csv-output", default=str(Path(__file__).with_name("benchmark_compact_framework.csv")))
    parser.add_argument("--applications", nargs="+", choices=APPLICATIONS, default=list(APPLICATIONS))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--numeric-direct-ids",
        action="store_true",
        help="skip compact vertex remapping for numeric streams; trades sparse vertex arrays for no Python ID map",
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    input_path = Path(args.input)
    records: List[Dict[str, Any]] = []
    prefixes = sorted(set(args.prefixes))
    for repetition in range(1, args.repetitions + 1):
        for prefix in prefixes:
            for application in args.applications:
                print(f"[{repetition}/{args.repetitions}] {application}, prefix={prefix}", flush=True)
                result = invoke(
                    input_path,
                    prefix,
                    application,
                    args.root,
                    args.duration,
                    args.numeric_direct_ids,
                )
                result["prefix_edges"] = prefix
                result["root"] = args.root
                result["duration"] = args.duration
                result["repetition"] = repetition
                records.append(result)
    payload = {
        "metadata": {
            "input": str(input_path.resolve()),
            "root": args.root,
            "duration": args.duration,
            "repetitions": args.repetitions,
            "applications": list(args.applications),
            "platform": platform.platform(),
            "python": sys.version,
            "representation": "NumPy numeric arrays + CSR endpoint-time channels; constant duration/weight compression; names discarded",
            "optimizations": "array-local reachability scan; FIFO monotone arrivals; Pareto pruning for pending shortest/fastest labels",
            "numeric_direct_ids": args.numeric_direct_ids,
        },
        "records": records,
    }
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if records:
        fieldnames = sorted({key for record in records for key in record})
        with Path(args.csv_output).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
