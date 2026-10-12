#!/usr/bin/env python3
"""Multi-root exact quality study for weighted temporal MST algorithms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from benchmark_weighted_mst import induced_temporal_sample
from temporal_mst_exact import exact_mst_w
from temporal_mst_ilp import ikuta_ilp
from temporal_mst_lns import shared_cost_lns
from temporal_mst_prototype import TemporalGraph, temporal_reachable
from temporal_mst_shared import shared_prefix_optimize, shortest_path_union_tree


def ranked_roots(
    graph: TemporalGraph,
    *,
    candidate_count: int,
    root_count: int,
) -> List[Tuple[str, int]]:
    outdegree = Counter(edge.u for edge in graph.edges)
    candidates = [vertex for vertex, _ in outdegree.most_common(candidate_count)]
    scored = [
        (vertex, len(temporal_reachable(graph, vertex, graph.edges)))
        for vertex in candidates
    ]
    scored.sort(key=lambda item: (-item[1], -outdegree[item[0]], item[0]))
    return scored[:root_count]


def timed(callable_object):
    started = time.perf_counter()
    result = callable_object()
    return result, time.perf_counter() - started


def evaluate_sample(
    dataset: str,
    graph: TemporalGraph,
    root: str,
    sample_vertices: int,
    time_limit: float,
) -> dict:
    sample = induced_temporal_sample(graph, root, sample_vertices)
    path_result, path_time = timed(lambda: shortest_path_union_tree(sample, root))
    path_cost = path_result[0]
    shared, shared_time = timed(
        lambda: shared_prefix_optimize(
            sample,
            root,
            max_rounds=2,
            max_hub_states=32,
            max_states_per_vertex=4,
            evaluated_prefixes=8,
        )
    )
    lns, lns_time = timed(
        lambda: shared_cost_lns(
            sample,
            root,
            max_rounds=2,
            max_hub_states=24,
            max_states_per_vertex=4,
            max_neighborhood=min(9, max(2, len(sample.vertices) - 1)),
            max_dp_cells=4_000_000,
        )
    )
    exact, exact_time = timed(
        lambda: exact_mst_w(
            sample,
            root,
            backend="implicit",
            max_terminals=max(12, len(sample.vertices) - 1),
            max_dp_cells=8_000_000,
        )
    )
    ilp, ilp_time = timed(
        lambda: ikuta_ilp(sample, root, time_limit=time_limit)
    )
    if not math.isclose(exact.cost, ilp.cost, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError("subset DP and Ikuta ILP disagree")
    optimum = exact.cost

    def ratio(cost: float) -> float:
        return cost / optimum if optimum > 0 else 1.0

    return {
        "dataset": dataset,
        "root": root,
        "requested_vertices": sample_vertices,
        "vertices": len(sample.vertices),
        "edges": len(sample.edges),
        "optimum": optimum,
        "path_union_cost": path_cost,
        "path_union_ratio": ratio(path_cost),
        "exchange_only_cost": shared.exchange_only_cost,
        "exchange_only_ratio": ratio(shared.exchange_only_cost),
        "shared_batch_cost": shared.shared_only_cost,
        "shared_batch_ratio": ratio(shared.shared_only_cost),
        "shared_combined_cost": shared.cost,
        "shared_combined_ratio": ratio(shared.cost),
        "lns_cost": lns.cost,
        "lns_ratio": ratio(lns.cost),
        "path_union_time_s": path_time,
        "shared_time_s": shared_time,
        "lns_time_s": lns_time,
        "subset_dp_time_s": exact_time,
        "ikuta_ilp_time_s": ilp_time,
        "ikuta_rounds": ilp.rounds,
        "ikuta_cuts": ilp.cuts_added,
    }


def summarize(records: Sequence[dict]) -> Dict[str, dict]:
    summary = {}
    methods = (
        "path_union",
        "exchange_only",
        "shared_batch",
        "shared_combined",
        "lns",
    )
    for method in methods:
        ratios = [record[f"{method}_ratio"] for record in records]
        summary[method] = {
            "mean_ratio": statistics.fmean(ratios),
            "median_ratio": statistics.median(ratios),
            "max_ratio": max(ratios),
            "optimal_count": sum(math.isclose(value, 1.0, abs_tol=1e-9) for value in ratios),
            "instance_count": len(ratios),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--candidate-roots", type=int, default=24)
    parser.add_argument("--roots", type=int, default=6)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument(
        "--output", type=Path, default=Path("weighted_quality_results.json")
    )
    args = parser.parse_args()
    records = []
    root_metadata = {}
    for path in args.inputs:
        graph = TemporalGraph.from_json(str(path))
        roots = ranked_roots(
            graph,
            candidate_count=args.candidate_roots,
            root_count=args.roots,
        )
        root_metadata[path.stem] = roots
        for root, full_reachable in roots:
            for size in args.sizes:
                record = evaluate_sample(path.stem, graph, root, size, args.time_limit)
                record["full_reachable"] = full_reachable
                records.append(record)
                print(
                    f"{path.stem} root={root} n={record['vertices']} "
                    f"path={record['path_union_ratio']:.3f} "
                    f"batch={record['shared_batch_ratio']:.3f} "
                    f"lns={record['lns_ratio']:.3f}",
                    flush=True,
                )
    payload = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "roots": root_metadata,
            "sizes": args.sizes,
        },
        "summary": summarize(records),
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    csv_path = args.output.with_suffix(".csv")
    fieldnames = sorted(records[0])
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
