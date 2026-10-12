#!/usr/bin/env python3
"""Audit frozen measurements and produce descriptive, not inferential, summaries."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(path):
    payload = json.loads(path.read_text())
    metadata = payload["metadata"]
    configuration = metadata.get("configuration", metadata.get("config"))
    expected = sum(len(entry["roots"]) for entry in configuration["datasets"]) * len(configuration["methods"]) * configuration.get("repeats", 3)
    if len(payload["records"]) != expected:
        raise ValueError(f"{path}: expected {expected} runs, found {len(payload['records'])}")
    changed = []
    for name, digest in metadata["sha256"].items():
        source = Path(__file__).parent / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            changed.append(name)
    for name, digest in metadata.get("input_sha256", {}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            changed.append(name)
    groups = defaultdict(list)
    for record in payload["records"]:
        groups[(record["dataset"], record["root"], record["method"])].append(record)
    summaries = []
    for (dataset, root, method), records in sorted(groups.items()):
        feasible = [record for record in records if record.get("validated")]
        row = {"result_file": path.name, "dataset": dataset, "root": root, "method": method, "runs": len(records), "feasible_runs": len(feasible), "optimal_runs": sum(record.get("certified_optimal", record.get("optimal", False)) for record in feasible), "over_budget_runs": sum(record.get("deadline_exceeded", False) for record in records), "hashes_match": not changed, "statuses": sorted(set(record.get("status", "feasible") for record in records))}
        for key in ("cost", "lower_bound", "wall_time_s", "peak_rss_mib", "safe_edges", "original_edges", "pool_edges", "root_cuts", "deadline_cuts"):
            values = [record[key] for record in records if key in record]
            if values:
                row[key] = {"median": statistics.median(values), "min": min(values), "max": max(values), "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}
        gaps = [max(0.0, (record["cost"] - record["lower_bound"]) / record["cost"]) if record["cost"] else 0.0 for record in feasible if "lower_bound" in record]
        if gaps:
            row["relative_gap"] = {"median": statistics.median(gaps), "min": min(gaps), "max": max(gaps)}
        summaries.append(row)
    return {"result_file": path.name, "expected_runs": expected, "hashes_match": not changed, "changed_files": changed, "summaries": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, default=Path("work/research_2026_10_07/summary"))
    args = parser.parse_args()
    audits = [summarize(path) for path in args.results]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps({"audits": audits}, indent=2, allow_nan=False) + "\n")
    fields = ["result_file", "dataset", "root", "method", "runs", "feasible_runs", "optimal_runs", "hashes_match", "median_cost", "median_lower_bound", "median_gap_pct", "median_wall_s", "min_wall_s", "max_wall_s", "max_rss_mib", "over_budget_runs"]
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            for row in audit["summaries"]:
                flat = {key: row[key] for key in fields if key in row}
                for target, source, statistic in (("median_cost", "cost", "median"), ("median_lower_bound", "lower_bound", "median"), ("median_gap_pct", "relative_gap", "median"), ("median_wall_s", "wall_time_s", "median"), ("min_wall_s", "wall_time_s", "min"), ("max_wall_s", "wall_time_s", "max"), ("max_rss_mib", "peak_rss_mib", "max")):
                    if source in row:
                        flat[target] = row[source][statistic] * (100 if source == "relative_gap" else 1)
                writer.writerow(flat)
            print(audit["result_file"], audit["expected_runs"], "runs, hashes match:", audit["hashes_match"])


if __name__ == "__main__":
    main()
