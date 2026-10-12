#!/usr/bin/env python3
"""Fresh-process warm-start fairness audit, separate from earlier cold runs."""

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


def worker(args):
    from temporal_mst_certificate import build_prefix_model, solve_prefix_lp
    from temporal_mst_highs import ikuta_warm_start
    from temporal_mst_priced import priced_warm_optimize
    from temporal_mst_prototype import TemporalGraph
    from temporal_mst_shared import shortest_path_union_tree

    started = time.monotonic()
    graph = TemporalGraph.from_json(str(args.input))
    if args.method.startswith("warm_closure_"):
        result = priced_warm_optimize(graph, args.root, time_limit=max(1e-6, args.budget - (time.monotonic() - started)), cut_mode=args.method[len("warm_closure_"):], max_probes=args.probes)
        baseline = result["baseline_cost"]
        result.pop("reachable")
    else:
        baseline, tree, _ = shortest_path_union_tree(graph, args.root)
        cuts = ()
        if args.method == "warm_precut":
            lp = solve_prefix_lp(build_prefix_model(graph, args.root), cut_mode="root", max_probes=args.probes, max_rounds=3, time_limit=args.budget * 0.25)
            cuts = lp.cuts
        remaining = max(1e-6, args.budget - (time.monotonic() - started))
        result = ikuta_warm_start(graph, args.root, time_limit=remaining, initial_ids=[edge.eid for edge in tree], cuts=cuts)
        result["root_cuts"] = len(cuts)
    ids = result.pop("edge_ids")
    elapsed = time.monotonic() - started
    result.update(baseline_cost=baseline, root=args.root, vertices=len(graph.vertices), edges=len(graph.edges), wall_time_s=elapsed, method=args.method, budget_s=args.budget, deadline_exceeded=elapsed > args.budget, validated=True, solution_sha256=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_mib"] = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--root")
    parser.add_argument("--method", choices=("warm", "warm_precut", "warm_closure_weight", "warm_closure_none", "warm_closure_root", "warm_closure_deadline"), default="warm")
    parser.add_argument("--probes", type=int, default=6)
    parser.add_argument("--budget", type=float, default=30)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("warm_start_results.json"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    config = json.loads(args.config.read_text())
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in ("temporal_mst_highs.py", "temporal_mst_priced.py", "temporal_mst_certificate.py", "temporal_mst_ilp.py", "temporal_mst_shared.py", "temporal_mst_prototype.py")]]
    import highspy
    import numpy
    import scipy
    metadata = {"date": time.strftime("%Y-%m-%d"), "config": config, "python": sys.version, "platform": platform.platform(), "versions": {"highspy": highspy.Highs().version(), "scipy": scipy.__version__, "numpy": numpy.__version__}, "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, "input_sha256": {entry["input"]: hashlib.sha256(Path(entry["input"]).read_bytes()).hexdigest() for entry in config["datasets"]}, "fresh_process": True, "threads": 1, "timing_scope": "input + initial tree + optional LP separation + model + native solver + validation; imports excluded"}
    records = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for entry in config["datasets"]:
        for root in entry["roots"]:
            for method in config.get("methods", ("warm", "warm_precut", "warm_closure_none", "warm_closure_root", "warm_closure_deadline")):
                for repeat in range(config.get("repeats", 3)):
                    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--input", entry["input"], "--root", root, "--budget", str(config["budget_s"]), "--method", method, "--probes", str(config.get("probes", 6))]
                    deadline = 2 * config["budget_s"] + 20
                    try:
                        process = subprocess.run(command, capture_output=True, text=True, env=env, timeout=deadline)
                        if process.returncode:
                            raise RuntimeError(process.stderr)
                        record = json.loads(process.stdout)
                    except subprocess.TimeoutExpired:
                        record = {"root": root, "method": method, "status": "hard-timeout", "validated": False, "wall_time_s": deadline}
                    record.update(dataset=entry["name"], repeat=repeat)
                    records.append(record)
                    args.output.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2, allow_nan=False) + "\n")
                    gap = record.get("relative_gap", record.get("mip_gap", 0.0))
                    print(f"{entry['name']} root={root} {method} repeat={repeat} cost={record.get('cost')} gap={gap:.4f} time={record['wall_time_s']:.3f}", flush=True)


if __name__ == "__main__":
    main()
