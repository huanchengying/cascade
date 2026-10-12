#!/usr/bin/env python3
"""Audit the 100K-event SpeedSteiner baseline against cascade incumbents."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_cascade_results import audit_hashes


ROW_FIELDS = (
    "dataset",
    "root",
    "status",
    "available",
    "speedsteiner_cost",
    "cascade_cost",
    "cost_ratio",
    "outcome",
    "wall_time_s",
    "peak_rss_mib",
    "state_count",
    "terminals",
)


def _aggregate(label: str, rows: list[dict]) -> dict:
    available = [row for row in rows if row["available"]]
    ratios = [row["cost_ratio"] for row in available]
    statuses = defaultdict(int)
    for row in rows:
        statuses[row["status"]] += 1
    return {
        "dataset": label,
        "units": len(rows),
        "available_units": len(available),
        "statuses": dict(sorted(statuses.items())),
        "wins": sum(row["outcome"] == "win" for row in available),
        "ties": sum(row["outcome"] == "tie" for row in available),
        "losses": sum(row["outcome"] == "loss" for row in available),
        "geometric_mean_cost_ratio": (
            math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
            if ratios
            else None
        ),
        "median_wall_s": (
            statistics.median(row["wall_time_s"] for row in available)
            if available
            else None
        ),
        "median_peak_rss_mib": (
            statistics.median(row["peak_rss_mib"] for row in available)
            if available
            else None
        ),
    }


def summarize(speed_payload: dict, cascade_payload: dict) -> tuple[list[dict], list[dict]]:
    configuration = speed_payload["metadata"]["config"]
    expected_units = [
        (entry["name"], str(root))
        for entry in configuration["datasets"]
        for root in entry["roots"]
    ]
    speed_by_unit = {}
    for record in speed_payload["records"]:
        key = record["dataset"], str(record["root"])
        if key in speed_by_unit:
            raise ValueError(f"duplicate SpeedSteiner unit: {key}")
        speed_by_unit[key] = record
    cascade_by_unit = {}
    for record in cascade_payload["records"]:
        if record["method"] != "cascade":
            continue
        key = record["dataset"], str(record["root"])
        if key in cascade_by_unit:
            raise ValueError(f"duplicate cascade unit: {key}")
        cascade_by_unit[key] = record
    if set(speed_by_unit) != set(expected_units):
        raise ValueError("SpeedSteiner records do not match configured dataset-root units")
    if not set(expected_units) <= set(cascade_by_unit):
        raise ValueError("cascade comparison is missing configured dataset-root units")

    rows = []
    for dataset, root in expected_units:
        speed = speed_by_unit[dataset, root]
        cascade = cascade_by_unit[dataset, root]
        if not cascade.get("validated", False):
            raise ValueError(f"unvalidated cascade result: {(dataset, root)}")
        available = bool(speed.get("validated", False))
        row = {
            "dataset": dataset,
            "root": root,
            "status": speed.get("status", "feasible"),
            "available": available,
            "speedsteiner_cost": speed.get("cost"),
            "cascade_cost": float(cascade["cost"]),
            "cost_ratio": None,
            "outcome": None,
            "wall_time_s": float(speed["wall_time_s"]),
            "peak_rss_mib": float(speed["peak_rss_mib"]),
            "state_count": speed.get("state_count"),
            "terminals": speed.get("terminals"),
        }
        if available:
            cost = float(speed["cost"])
            cascade_cost = float(cascade["cost"])
            tolerance = 1e-9 * max(1.0, abs(cost), abs(cascade_cost))
            row["cost_ratio"] = cost / cascade_cost if cascade_cost else 1.0
            row["outcome"] = (
                "win"
                if cost + tolerance < cascade_cost
                else "loss"
                if cost > cascade_cost + tolerance
                else "tie"
            )
        rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    summaries = [_aggregate(name, values) for name, values in sorted(grouped.items())]
    summaries.append(_aggregate("ALL", rows))
    return rows, summaries


def compare_matched_budget(speed_payload: dict, budget_payload: dict) -> list[dict]:
    speed_by_unit = {
        (record["dataset"], str(record["root"])): record
        for record in speed_payload["records"]
    }
    configuration = budget_payload["metadata"]["config"]
    comparison_rows = []
    for method in configuration["methods"]:
        available = []
        for record in budget_payload["records"]:
            if record["method"] != method:
                continue
            speed = speed_by_unit.get((record["dataset"], str(record["root"])))
            if speed is None or record["budget_s"] != speed["budget_s"]:
                continue
            if not speed.get("validated") or not record.get("validated"):
                continue
            ratio = float(speed["cost"]) / float(record["cost"])
            tolerance = 1e-9 * max(1.0, abs(speed["cost"]), abs(record["cost"]))
            available.append(
                {
                    "dataset": record["dataset"],
                    "root": record["root"],
                    "budget_s": record["budget_s"],
                    "speedsteiner_cost": speed["cost"],
                    "comparison_cost": record["cost"],
                    "cost_ratio": ratio,
                    "speedsteiner_wall_s": speed["wall_time_s"],
                    "comparison_wall_s": record["wall_time_s"],
                    "outcome": (
                        "win" if speed["cost"] + tolerance < record["cost"]
                        else "loss" if speed["cost"] > record["cost"] + tolerance
                        else "tie"
                    ),
                }
            )
        comparison_rows.append(
            {
                "comparison_method": method,
                "paired_units": len(available),
                "wins": sum(row["outcome"] == "win" for row in available),
                "ties": sum(row["outcome"] == "tie" for row in available),
                "losses": sum(row["outcome"] == "loss" for row in available),
                "geometric_mean_cost_ratio": (
                    math.exp(statistics.fmean(math.log(row["cost_ratio"]) for row in available))
                    if available else None
                ),
                "comparison_over_budget_units": sum(
                    row["comparison_wall_s"] > row["budget_s"] for row in available
                ),
                "pairs": available,
            }
        )
    return comparison_rows


def _markdown(summaries: list[dict], hashes_match: bool) -> str:
    lines = [
        "# 100K-event SpeedSteiner audit",
        "",
        f"Frozen source and input hashes match: **{'yes' if hashes_match else 'no'}**.",
        "SpeedSteiner has a 10-second per-run budget. Cost ratios use Cascade after up to 32 star rounds and four cascade rounds as a quality reference; this is not an equal-time comparison and the star round cap does not establish convergence.",
        "",
        "| Dataset | Available | Statuses | W/T/L vs Cascade | Geomean ratio | Median time | Median RSS |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        statuses = ", ".join(f"{key}:{value}" for key, value in summary["statuses"].items())
        ratio = summary["geometric_mean_cost_ratio"]
        wall = summary["median_wall_s"]
        rss = summary["median_peak_rss_mib"]
        lines.append(
            f"| {summary['dataset']} | {summary['available_units']}/{summary['units']} | "
            f"{statuses} | {summary['wins']}/{summary['ties']}/{summary['losses']} | "
            f"{ratio:.6f} | {wall:.3f} s | {rss:.1f} MiB |"
            if ratio is not None
            else f"| {summary['dataset']} | 0/{summary['units']} | {statuses} | 0/0/0 | -- | -- | -- |"
        )
    lines.extend(
        [
            "",
            "Guarded units remain in the denominator. A size guard means the temporal-state count times the terminal count exceeded the adapter's frozen 50,000,000 limit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed-results", type=Path, required=True)
    parser.add_argument("--cascade-results", type=Path, required=True)
    parser.add_argument("--budget-results", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("work/research_2026_10_08/speedsteiner_100k_audit"),
    )
    args = parser.parse_args()
    speed_payload = json.loads(args.speed_results.read_text())
    cascade_payload = json.loads(args.cascade_results.read_text())
    budget_payload = json.loads(args.budget_results.read_text()) if args.budget_results else None
    checks = audit_hashes(speed_payload, Path.cwd()) + audit_hashes(
        cascade_payload, Path.cwd()
    )
    if budget_payload is not None:
        from verify_artifact_20261008 import audit_results

        audit_results(args.budget_results.resolve(), Path.cwd())
        checks += audit_hashes(budget_payload, Path.cwd())
    hashes_match = all(check["matches"] for check in checks)
    if not hashes_match:
        changed = [check["path"] for check in checks if not check["matches"]]
        raise ValueError(f"frozen hashes no longer match: {changed}")
    rows, summaries = summarize(speed_payload, cascade_payload)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "speed_results": str(args.speed_results),
                "cascade_results": str(args.cascade_results),
                "hash_checks": checks,
                "rows": rows,
                "summaries": summaries,
                "matched_budget_comparisons": (
                    compare_matched_budget(speed_payload, budget_payload)
                    if budget_payload is not None else []
                ),
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
        _markdown(summaries, hashes_match)
    )
    print(f"wrote {args.output_prefix}.[json,csv,md]")


if __name__ == "__main__":
    main()
