#!/usr/bin/env python3
"""Freeze alternative trust semantics and measured daily mobility workloads."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from prepare_research_workloads import transform_payload
from temporal_mst_prototype import TemporalGraph, temporal_reachable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("work/research_2026_10_07/sensitivity"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    sources = [("alpha", Path("work/data/processed/bitcoin-alpha.json"), "10"), ("otc", Path("work/data/processed/bitcoin-otc.json"), "13")]
    for name, source, root in sources:
        raw = source.read_bytes()
        payload = json.loads(raw)
        for policy in ("trust-positive", "trust-exponential"):
            result = transform_payload(payload, policy)
            result["metadata"].update(parent_input=str(source), parent_sha256=hashlib.sha256(raw).hexdigest())
            destination = args.output_dir / (name + "-" + policy + ".json")
            destination.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
            datasets.append({"name": name + "-" + policy, "input": str(destination), "roots": [root]})
    source = Path("work/data/processed/citibike-202604-distance-100k.json")
    raw = source.read_bytes()
    payload = json.loads(raw)
    for day in range(2):
        window = (day * 86400.0, (day + 1) * 86400.0)
        window_payload = transform_payload(payload, "recorded-duration", window)
        counts = Counter(edge["u"] for edge in window_payload["edges"])
        roots = sorted(counts, key=lambda root: (-counts[root], root))[:10]
        for policy in ("geographic-distance", "recorded-duration"):
            result = transform_payload(payload, policy, window)
            result["metadata"].update(parent_input=str(source), parent_sha256=hashlib.sha256(raw).hexdigest())
            destination = args.output_dir / ("citi-day" + str(day + 1) + "-" + policy + ".json")
            destination.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
            graph = TemporalGraph.from_json(str(destination))
            root = min(roots, key=lambda candidate: (-len(temporal_reachable(graph, candidate, graph.edges)), candidate))
            result["metadata"].update(root_selection="maximum temporal reachability among 10 highest event-outdegree roots; lexical ties", selected_root=root)
            destination.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
            datasets.append({"name": destination.stem, "input": str(destination), "roots": [root]})
    configuration = {"datasets": datasets, "methods": ["warm", "warm_closure_weight", "warm_closure_deadline"], "budget_s": 10, "probes": 6, "repeats": 3}
    (args.output_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    print(json.dumps(configuration, indent=2))


if __name__ == "__main__":
    main()
