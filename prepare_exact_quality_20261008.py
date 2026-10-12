#!/usr/bin/env python3
"""Freeze the expanded real-data exact sample suite for the October 8 audit."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from benchmark_weighted_mst import induced_temporal_sample
from temporal_mst_exact import exact_mst_w
from temporal_mst_ilp import ikuta_ilp
from temporal_mst_prototype import TemporalGraph


DATASETS = [
    ("alpha", "work/data/processed/bitcoin-alpha.json", ["10", "1"]),
    ("otc", "work/data/processed/bitcoin-otc.json", ["13", "7"]),
    (
        "citi-distance",
        "work/data/processed/citibike-202604-distance-100k.json",
        ["6197.08"],
    ),
    (
        "citi-duration",
        "work/research_2026_10_08/data/citibike-duration-100k.json",
        ["6197.08"],
    ),
    (
        "tlc-distance",
        "work/data/processed/nyc-tlc-yellow-2025-01-distance-100k.json",
        ["161", "237"],
    ),
    (
        "tlc-duration",
        "work/data/processed/nyc-tlc-yellow-2025-01-duration-100k.json",
        ["161", "237"],
    ),
    (
        "tlc-total-amount",
        "work/data/processed/nyc-tlc-yellow-2025-01-total-amount-100k.json",
        ["161", "237"],
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/research_2026_10_08/exact_samples"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances = []
    optima = []
    for name, source, roots in DATASETS:
        graph = TemporalGraph.from_json(source)
        checksum = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        for root in roots:
            for size in (8, 10):
                sample = induced_temporal_sample(graph, root, size)
                exact = exact_mst_w(sample, root, max_dp_cells=8_000_000)
                ilp = ikuta_ilp(sample, root, time_limit=10)
                if abs(exact.cost - ilp.cost) > 1e-6:
                    raise RuntimeError("independent exact oracles disagree")
                key = f"{name}-root{root}-n{size}"
                path = args.output_dir / f"{key}.json"
                payload = {
                    "metadata": {
                        "parent_input": source,
                        "parent_sha256": checksum,
                        "sampling": (
                            "earliest-arrival-tree closed vertex-induced sample"
                        ),
                        "root": root,
                        "requested_vertices": size,
                    },
                    "vertices": sample.vertices,
                    "edges": [
                        {
                            "id": edge.eid,
                            "u": edge.u,
                            "v": edge.v,
                            "start": edge.start,
                            "arrival": edge.arrival,
                            "weight": edge.weight,
                        }
                        for edge in sample.edges
                    ],
                }
                path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
                instances.append({"name": key, "input": str(path), "roots": [root]})
                optima.append(
                    {
                        "name": key,
                        "cost": exact.cost,
                        "vertices": len(sample.vertices),
                        "edges": len(sample.edges),
                        "subset_dp": asdict(exact),
                        "ikuta": asdict(ilp),
                    }
                )
    (args.output_dir / "optima.json").write_text(
        json.dumps(optima, indent=2, allow_nan=False) + "\n"
    )
    configuration = {
        "datasets": instances,
        "root_selection": {
            "provenance": "work/research_2026_10_08/root_selection_20261008.json",
            "policy": "first selected Citi root and first two selected TLC roots; Bitcoin roots frozen by the October 7 exact suite",
        },
        "methods": [
            "path_union",
            "star",
            "cascade",
            "huang2",
            "huang3",
            "ikuta",
            "closure_root",
        ],
        "budget_s": 15,
        "probes": 12,
        "star_rounds": 32,
        "cascade_rounds": 4,
        "cascade_max_hubs": 10000,
        "cascade_max_parent_evaluations": 100000,
        "repeats": 3,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )
    print(f"froze {len(instances)} independent-oracle samples in {args.output_dir}")


if __name__ == "__main__":
    main()
