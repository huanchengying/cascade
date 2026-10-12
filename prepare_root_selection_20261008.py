#!/usr/bin/env python3
"""Freeze algorithm-independent root selection for the 100K-event study."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from temporal_mst_prototype import TemporalGraph, temporal_reachable


DATASETS = (
    (
        "citibike-100k",
        "work/data/processed/citibike-202604-distance-100k.json",
    ),
    (
        "nyc-tlc-100k",
        "work/data/processed/nyc-tlc-yellow-2025-01-duration-100k.json",
    ),
)


def select_roots(graph, candidate_count: int = 20, selection_count: int = 4):
    if not 0 < selection_count <= candidate_count:
        raise ValueError("root counts must satisfy 0 < selection <= candidates")
    outdegree = Counter(edge.u for edge in graph.edges)
    candidates = sorted(
        graph.vertices,
        key=lambda vertex: (-outdegree[vertex], str(vertex)),
    )[:candidate_count]
    rows = [
        {
            "root": str(root),
            "temporal_reachability": len(temporal_reachable(graph, root, graph.edges)),
            "event_outdegree": outdegree[root],
        }
        for root in candidates
    ]
    rows.sort(
        key=lambda row: (
            -row["temporal_reachability"],
            -row["event_outdegree"],
            row["root"],
        )
    )
    return rows[:selection_count], rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--selection-count", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/research_2026_10_08/root_selection_20261008.json"),
    )
    args = parser.parse_args()
    datasets = []
    for name, input_path in DATASETS:
        graph = TemporalGraph.from_json(input_path)
        selected, ranking = select_roots(
            graph,
            candidate_count=args.candidate_count,
            selection_count=args.selection_count,
        )
        datasets.append(
            {
                "name": name,
                "input": input_path,
                "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                "vertices": len(graph.vertices),
                "events": len(graph.edges),
                "selected_roots": [row["root"] for row in selected],
                "candidate_ranking": ranking,
            }
        )
    payload = {
        "generator": Path(__file__).name,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rule": {
            "candidate_count": args.candidate_count,
            "selection_count": args.selection_count,
            "candidate_order": "descending event outdegree, then vertex ID",
            "selection_order": "descending temporal reachability, descending event outdegree, then vertex ID",
            "performance_blind": True,
            "registration_scope": "frozen before final reruns; exploratory outcomes were already known, so not an independent prospective preregistration",
        },
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
