#!/usr/bin/env python3
"""Summarize exact-sample approximation quality against frozen dual oracles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_cascade_results import audit_hashes


SAMPLE_PATTERN = re.compile(r"^(?P<family>.+)-root(?P<root>.+)-n(?P<size>[0-9]+)$")
ROW_FIELDS = (
    "dataset",
    "family",
    "root",
    "vertices",
    "method",
    "repeats",
    "feasible_repeats",
    "available",
    "statuses",
    "cost",
    "optimum",
    "cost_ratio",
    "relative_gap_pct",
    "hits_optimum",
    "median_wall_s",
)


def _sample_parts(name: str) -> tuple[str, str, int]:
    match = SAMPLE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"unrecognized exact sample name: {name}")
    return match.group("family"), match.group("root"), int(match.group("size"))


def summarize_exact(payload: dict, optima: list[dict]) -> tuple[list[dict], list[dict], dict]:
    optimum_by_name = {entry["name"]: float(entry["cost"]) for entry in optima}
    if len(optimum_by_name) != len(optima):
        raise ValueError("duplicate exact sample names")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in payload["records"]:
        groups[(record["dataset"], record["method"])].append(record)

    configuration = payload["metadata"].get(
        "configuration", payload["metadata"].get("config", {})
    )
    methods = configuration.get("methods", sorted({method for _, method in groups}))
    datasets = [entry["name"] for entry in configuration.get("datasets", [])]
    if not datasets:
        datasets = sorted({dataset for dataset, _ in groups})
    if set(datasets) != set(optimum_by_name):
        missing = sorted(set(datasets) ^ set(optimum_by_name))
        raise ValueError(f"result/optimum sample mismatch: {missing}")

    rows = []
    for dataset in datasets:
        family, root, size = _sample_parts(dataset)
        optimum = optimum_by_name[dataset]
        for method in methods:
            records = groups.get((dataset, method), [])
            if not records:
                raise ValueError(f"missing method {method} for {dataset}")
            feasible = [record for record in records if record.get("validated", False)]
            statuses = sorted({record.get("status", "feasible") for record in records})
            if feasible and len(feasible) != len(records):
                raise ValueError(f"inconsistent repeat availability for {dataset}: {method}")
            if not feasible:
                rows.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "root": root,
                        "vertices": size,
                        "method": method,
                        "repeats": len(records),
                        "feasible_repeats": 0,
                        "available": False,
                        "statuses": statuses,
                        "cost": None,
                        "optimum": optimum,
                        "cost_ratio": None,
                        "relative_gap_pct": None,
                        "hits_optimum": False,
                        "median_wall_s": statistics.median(
                            float(record["wall_time_s"]) for record in records
                        ),
                    }
                )
                continue
            costs = [float(record["cost"]) for record in feasible]
            cost = statistics.median(costs)
            tolerance = 1e-7 * max(1.0, abs(optimum), abs(cost))
            if cost + tolerance < optimum:
                raise ValueError(f"reported cost below frozen optimum for {dataset}: {method}")
            ratio = cost / optimum if optimum else 1.0
            rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "root": root,
                    "vertices": size,
                    "method": method,
                    "repeats": len(records),
                    "feasible_repeats": len(feasible),
                    "available": True,
                    "statuses": statuses,
                    "cost": cost,
                    "optimum": optimum,
                    "cost_ratio": ratio,
                    "relative_gap_pct": 100.0 * (ratio - 1.0),
                    "hits_optimum": abs(cost - optimum) <= tolerance,
                    "median_wall_s": statistics.median(
                        float(record["wall_time_s"]) for record in feasible
                    ),
                }
            )

    summaries = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        available = [row for row in selected if row["available"]]
        ratios = [row["cost_ratio"] for row in available]
        summaries.append(
            {
                "method": method,
                "samples": len(selected),
                "available_samples": len(available),
                "unavailable_samples": len(selected) - len(available),
                "unavailable_statuses": sorted(
                    {
                        status
                        for row in selected
                        if not row["available"]
                        for status in row["statuses"]
                    }
                ),
                "mean_cost_ratio": statistics.fmean(ratios) if ratios else None,
                "geometric_mean_cost_ratio": (
                    math.exp(
                        statistics.fmean(
                            math.log(ratio) for ratio in ratios if ratio > 0
                        )
                    )
                    if ratios
                    else None
                ),
                "median_cost_ratio": statistics.median(ratios) if ratios else None,
                "worst_cost_ratio": max(ratios) if ratios else None,
                "optimal_hits": sum(row["hits_optimum"] for row in available),
                "median_wall_s": (
                    statistics.median(row["median_wall_s"] for row in available)
                    if available
                    else None
                ),
            }
        )

    by_sample_method = {
        (row["dataset"], row["method"]): row for row in rows
    }
    star_pairs = []
    for dataset in datasets:
        if (dataset, "star") not in by_sample_method or (
            dataset,
            "cascade",
        ) not in by_sample_method:
            continue
        star = by_sample_method[(dataset, "star")]
        cascade = by_sample_method[(dataset, "cascade")]
        if not star["available"] or not cascade["available"]:
            continue
        tolerance = 1e-9 * max(1.0, star["cost"], cascade["cost"])
        delta = star["cost"] - cascade["cost"]
        outcome = "win" if delta > tolerance else "loss" if delta < -tolerance else "tie"
        star_pairs.append(
            {
                "dataset": dataset,
                "star_cost": star["cost"],
                "cascade_cost": cascade["cost"],
                "outcome": outcome,
                "cascade_hits_optimum": cascade["hits_optimum"],
            }
        )
    comparison = {
        "paired_samples": len(star_pairs),
        "wins": sum(pair["outcome"] == "win" for pair in star_pairs),
        "ties": sum(pair["outcome"] == "tie" for pair in star_pairs),
        "losses": sum(pair["outcome"] == "loss" for pair in star_pairs),
        "wins_reaching_optimum": sum(
            pair["outcome"] == "win" and pair["cascade_hits_optimum"]
            for pair in star_pairs
        ),
        "pairs": star_pairs,
    }
    return rows, summaries, comparison


def _markdown(summaries: list[dict], comparison: dict, hashes_match: bool) -> str:
    lines = [
        "# Expanded exact-sample quality audit",
        "",
        f"Frozen benchmark source and input hashes match: **{'yes' if hashes_match else 'no'}**.",
        "Each optimum is independently confirmed by subset DP and the Ikuta edge/prefix ILP before benchmarking.",
        "Timing repeats are collapsed before quality statistics. Samples sharing a parent graph, root, or nested vertex set are not independent statistical replicates.",
        "",
        "## Method quality",
        "",
        "| Method | Available | Mean / OPT | Geomean / OPT | Worst / OPT | Optimal hits | Median time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['method']} | {summary['available_samples']}/{summary['samples']} | "
            f"{summary['mean_cost_ratio']:.6f} | "
            f"{summary['geometric_mean_cost_ratio']:.6f} | "
            f"{summary['worst_cost_ratio']:.6f} | "
            f"{summary['optimal_hits']}/{summary['available_samples']} | "
            f"{summary['median_wall_s']:.4f} s |"
        )
    unavailable = [summary for summary in summaries if summary["unavailable_samples"]]
    if unavailable:
        lines.extend(["", "Unavailable runs are retained explicitly:"])
        for summary in unavailable:
            lines.append(
                f"- `{summary['method']}`: {summary['unavailable_samples']} samples; "
                f"statuses {', '.join(summary['unavailable_statuses'])}."
            )
    lines.extend(
        [
            "",
            "## Cascade versus conservative star",
            "",
            f"Across {comparison['paired_samples']} paired samples, cascade has "
            f"{comparison['wins']} wins, {comparison['ties']} ties, and "
            f"{comparison['losses']} losses. "
            f"{comparison['wins_reaching_optimum']} strict wins reach the frozen optimum.",
            "",
            "This table measures small-instance solution quality, not scalability. The full Ikuta method is included as an exact reference and is not expected to lose on these sizes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--additional-results", type=Path, nargs="*", default=[])
    parser.add_argument("--optima", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("work/research_2026_10_08/exact_quality_summary"),
    )
    args = parser.parse_args()

    result_paths = [args.results, *args.additional_results]
    payloads = [json.loads(path.read_text()) for path in result_paths]
    payload = payloads[0]
    optima = json.loads(args.optima.read_text())
    primary_configuration = payload["metadata"].get(
        "configuration", payload["metadata"].get("config")
    )
    dataset_names = {entry["name"] for entry in primary_configuration["datasets"]}
    methods = list(primary_configuration["methods"])
    records = []
    hash_checks = []
    for path, result_payload in zip(result_paths, payloads):
        configuration = result_payload["metadata"].get(
            "configuration", result_payload["metadata"].get("config")
        )
        if {entry["name"] for entry in configuration["datasets"]} != dataset_names:
            raise ValueError(f"dataset mismatch in additional result file: {path}")
        expected = (
            sum(len(dataset["roots"]) for dataset in configuration["datasets"])
            * len(configuration["methods"])
            * configuration.get("repeats", 1)
        )
        if len(result_payload["records"]) != expected:
            raise ValueError(
                f"{path}: expected {expected} records, "
                f"found {len(result_payload['records'])}"
            )
        for method in configuration["methods"]:
            if method not in methods:
                methods.append(method)
        records.extend(result_payload["records"])
        for check in audit_hashes(result_payload, Path.cwd()):
            hash_checks.append({"result_file": str(path), **check})
    hashes_match = all(check["matches"] for check in hash_checks)
    if not hashes_match:
        changed = [check["path"] for check in hash_checks if not check["matches"]]
        raise ValueError(f"frozen hashes no longer match: {changed}")

    combined = {
        "metadata": {
            "configuration": {
                **primary_configuration,
                "methods": methods,
            }
        },
        "records": records,
    }
    rows, summaries, comparison = summarize_exact(combined, optima)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "sources": [str(path) for path in result_paths],
                "optima": str(args.optima),
                "hash_checks": hash_checks,
                "rows": rows,
                "method_summaries": summaries,
                "cascade_vs_star": comparison,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=ROW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    args.output_prefix.with_suffix(".md").write_text(
        _markdown(summaries, comparison, hashes_match)
    )
    print(
        f"summarized {len(rows)} method-sample rows; cascade/star "
        f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']}"
    )


if __name__ == "__main__":
    main()
