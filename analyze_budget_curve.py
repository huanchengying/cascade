#!/usr/bin/env python3
"""Audit matched-incumbent budget curves and mechanism ablations."""

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
    "budget_s",
    "method",
    "runs",
    "available",
    "statuses",
    "initial_cost",
    "cost",
    "improvement_from_initial_pct",
    "lower_bound",
    "relative_gap_pct",
    "wall_time_s",
    "over_budget_runs",
    "peak_rss_mib",
    "lp_time_s",
    "root_cuts",
    "safe_edges",
    "pool_edges",
    "failed_worker_attempts",
    "failed_attempt_wall_s",
)


def _median(records: list[dict], key: str, default=None):
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return statistics.median(values) if values else default


def _classify(left: float, right: float) -> str:
    tolerance = 1e-9 * max(1.0, abs(left), abs(right))
    return "win" if left + tolerance < right else "loss" if left > right + tolerance else "tie"


def summarize_budget(payload: dict) -> tuple[list[dict], list[dict], dict, list[dict]]:
    metadata = payload["metadata"]
    configuration = metadata.get("configuration", metadata.get("config", {}))
    methods = configuration["methods"]
    groups: dict[tuple[str, str, float, str], list[dict]] = defaultdict(list)
    for record in payload["records"]:
        groups[
            (
                record["dataset"],
                str(record["root"]),
                float(record["budget_s"]),
                record["method"],
            )
        ].append(record)

    rows = []
    unit_keys = sorted({key[:3] for key in groups})
    for dataset, root, budget in unit_keys:
        for method in methods:
            records = groups.get((dataset, root, budget, method), [])
            if not records:
                raise ValueError(
                    f"missing budget record: {dataset} root={root} "
                    f"budget={budget} method={method}"
                )
            feasible = [record for record in records if record.get("validated", False)]
            statuses = sorted({record.get("status", "feasible") for record in records})
            if feasible and len(feasible) != len(records):
                raise ValueError(
                    f"inconsistent repeat availability: {dataset} root={root} "
                    f"budget={budget} method={method}"
                )
            if not feasible:
                rows.append(
                    {
                        "dataset": dataset,
                        "root": root,
                        "budget_s": budget,
                        "method": method,
                        "runs": len(records),
                        "available": False,
                        "statuses": statuses,
                        "over_budget_runs": sum(
                            record.get("deadline_exceeded", False) for record in records
                        ),
                        "wall_time_s": _median(records, "wall_time_s"),
                        "failed_worker_attempts": sum(
                            len(record.get("worker_failures", [])) for record in records
                        ),
                        "failed_attempt_wall_s": math.fsum(
                            failure["wall_time_s"]
                            for record in records
                            for failure in record.get("worker_failures", [])
                        ),
                    }
                )
                continue
            initial = _median(feasible, "initial_cost")
            cost = _median(feasible, "cost")
            lower = _median(feasible, "lower_bound", 0.0)
            row = {
                "dataset": dataset,
                "root": root,
                "budget_s": budget,
                "method": method,
                "runs": len(records),
                "available": True,
                "statuses": statuses,
                "initial_cost": initial,
                "cost": cost,
                "improvement_from_initial_pct": (
                    100.0 * (initial - cost) / initial if initial else 0.0
                ),
                "lower_bound": lower,
                "relative_gap_pct": 100.0 * max(0.0, (cost - lower) / cost)
                if cost
                else 0.0,
                "wall_time_s": _median(feasible, "wall_time_s"),
                "over_budget_runs": sum(
                    record.get("deadline_exceeded", False) for record in records
                ),
                "peak_rss_mib": _median(feasible, "peak_rss_mib"),
                "lp_time_s": _median(feasible, "lp_time_s"),
                "root_cuts": _median(feasible, "root_cuts"),
                "safe_edges": _median(feasible, "safe_edges"),
                "pool_edges": _median(feasible, "pool_edges"),
                "failed_worker_attempts": sum(
                    len(record.get("worker_failures", [])) for record in records
                ),
                "failed_attempt_wall_s": math.fsum(
                    failure["wall_time_s"]
                    for record in records
                    for failure in record.get("worker_failures", [])
                ),
            }
            if lower > cost + 1e-7 * max(1.0, cost):
                raise ValueError(
                    f"global lower bound exceeds incumbent: {dataset} root={root} "
                    f"budget={budget} method={method}"
                )
            rows.append(row)

    by_unit: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        if row["available"]:
            by_unit[(row["dataset"], row["root"], row["budget_s"])].append(row)
    consistency = []
    for key, unit_rows in sorted(by_unit.items()):
        initial_costs = [row["initial_cost"] for row in unit_rows]
        spread = max(initial_costs) - min(initial_costs)
        tolerance = 1e-9 * max(1.0, max(abs(value) for value in initial_costs))
        consistency.append(
            {
                "dataset": key[0],
                "root": key[1],
                "budget_s": key[2],
                "methods_available": len(unit_rows),
                "initial_cost_min": min(initial_costs),
                "initial_cost_max": max(initial_costs),
                "initial_costs_match": spread <= tolerance,
            }
        )
    if not all(item["initial_costs_match"] for item in consistency):
        raise ValueError("methods did not receive the same deterministic incumbent cost")

    summaries = []
    budgets = [float(value) for value in configuration.get("budgets_s", [])]
    for budget in budgets:
        budget_rows = [row for row in rows if row["budget_s"] == budget]
        for method in methods:
            selected = [row for row in budget_rows if row["method"] == method]
            available = [row for row in selected if row["available"]]
            summaries.append(
                {
                    "budget_s": budget,
                    "method": method,
                    "units": len(selected),
                    "available_units": len(available),
                    "unavailable_statuses": sorted(
                        {
                            status
                            for row in selected if not row["available"]
                            for status in row["statuses"]
                        }
                    ),
                    "failed_worker_attempts": sum(
                        row["failed_worker_attempts"] for row in selected
                    ),
                    "improved_units": sum(
                        row["improvement_from_initial_pct"] > 1e-9 for row in available
                    ),
                    "mean_improvement_from_initial_pct": statistics.fmean(
                        row["improvement_from_initial_pct"] for row in available
                    )
                    if available
                    else None,
                    "geometric_mean_cost_to_initial": math.exp(
                        statistics.fmean(
                            math.log(row["cost"] / row["initial_cost"])
                            for row in available
                            if row["cost"] > 0 and row["initial_cost"] > 0
                        )
                    )
                    if available
                    else None,
                    "median_global_gap_pct": statistics.median(
                        row["relative_gap_pct"] for row in available
                    )
                    if available
                    else None,
                    "median_wall_s": statistics.median(
                        row["wall_time_s"] for row in available
                    )
                    if available
                    else None,
                    "max_wall_s": max(row["wall_time_s"] for row in available)
                    if available
                    else None,
                    "over_budget_runs": sum(row["over_budget_runs"] for row in selected),
                    "median_peak_rss_mib": statistics.median(
                        row["peak_rss_mib"] for row in available
                    )
                    if available
                    else None,
                }
            )

    row_index = {
        (row["dataset"], row["root"], row["budget_s"], row["method"]): row
        for row in rows
    }
    comparisons = {}
    for label, left_method, right_method in (
        (
            "root_pricing_vs_weight_pool",
            "cascade_closure_root",
            "cascade_closure_weight",
        ),
        ("full_root_precut_vs_full", "cascade_warm_root", "cascade_warm"),
        ("root_pricing_vs_full", "cascade_closure_root", "cascade_warm"),
    ):
        budget_results = []
        for budget in budgets:
            pairs = []
            for dataset, root, unit_budget in unit_keys:
                if unit_budget != budget:
                    continue
                left = row_index.get((dataset, root, budget, left_method))
                right = row_index.get((dataset, root, budget, right_method))
                if not left or not right or not left["available"] or not right["available"]:
                    continue
                outcome = _classify(left["cost"], right["cost"])
                pairs.append(
                    {
                        "dataset": dataset,
                        "root": root,
                        "left_cost": left["cost"],
                        "right_cost": right["cost"],
                        "relative_saving_pct": 100.0
                        * (right["cost"] - left["cost"])
                        / right["cost"],
                        "outcome": outcome,
                    }
                )
            budget_results.append(
                {
                    "budget_s": budget,
                    "paired_units": len(pairs),
                    "wins": sum(pair["outcome"] == "win" for pair in pairs),
                    "ties": sum(pair["outcome"] == "tie" for pair in pairs),
                    "losses": sum(pair["outcome"] == "loss" for pair in pairs),
                    "mean_relative_saving_pct": statistics.fmean(
                        pair["relative_saving_pct"] for pair in pairs
                    )
                    if pairs
                    else None,
                    "median_relative_saving_pct": statistics.median(
                        pair["relative_saving_pct"] for pair in pairs
                    )
                    if pairs
                    else None,
                    "pairs": pairs,
                }
            )
        comparisons[label] = {
            "left_method": left_method,
            "right_method": right_method,
            "budgets": budget_results,
        }

    monotonicity = []
    unit_without_budget = sorted({(dataset, root) for dataset, root, _ in unit_keys})
    for dataset, root in unit_without_budget:
        for method in methods:
            available = [
                row_index[(dataset, root, budget, method)]
                for budget in budgets
                if (dataset, root, budget, method) in row_index
                and row_index[(dataset, root, budget, method)]["available"]
            ]
            for earlier, later in zip(available, available[1:]):
                outcome = _classify(later["cost"], earlier["cost"])
                if outcome == "loss":
                    monotonicity.append(
                        {
                            "dataset": dataset,
                            "root": root,
                            "method": method,
                            "earlier_budget_s": earlier["budget_s"],
                            "later_budget_s": later["budget_s"],
                            "earlier_cost": earlier["cost"],
                            "later_cost": later["cost"],
                        }
                    )
    diagnostics = {
        "initial_consistency": consistency,
        "nonmonotone_budget_pairs": monotonicity,
        "failed_worker_attempts": [
            {
                "dataset": record["dataset"],
                "root": record["root"],
                "budget_s": record["budget_s"],
                "method": record["method"],
                **failure,
            }
            for record in payload["records"]
            for failure in record.get("worker_failures", [])
        ],
    }
    return rows, summaries, comparisons, diagnostics


def _markdown(summaries: list[dict], comparisons: dict, diagnostics: dict) -> str:
    lines = [
        "# Matched-incumbent budget curves",
        "",
        "Every method deterministically rebuilds the same cascade incumbent for a dataset-root-budget unit. Bounds shown below are full-graph LP or full-master bounds; restricted-master bounds are excluded.",
        "",
        "## Aggregate curve",
        "",
        "| Budget | Method | Available | Improved | Mean initial reduction | Median global gap | Median wall | Overruns | Median RSS |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        reduction = summary["mean_improvement_from_initial_pct"]
        gap = summary["median_global_gap_pct"]
        wall = summary["median_wall_s"]
        rss = summary["median_peak_rss_mib"]
        lines.append(
            f"| {summary['budget_s']:g} s | {summary['method']} | "
            f"{summary['available_units']}/{summary['units']} | "
            f"{summary['improved_units']} | "
            f"{f'{reduction:.3f}%' if reduction is not None else '--'} | "
            f"{f'{gap:.3f}%' if gap is not None else '--'} | "
            f"{f'{wall:.3f} s' if wall is not None else '--'} | "
            f"{summary['over_budget_runs']} | "
            f"{f'{rss:.1f} MiB' if rss is not None else '--'} |"
        )
    lines.extend(["", "## Paired mechanism comparisons", ""])
    for label, comparison in comparisons.items():
        lines.append(
            f"`{label}` compares `{comparison['left_method']}` against "
            f"`{comparison['right_method']}`:"
        )
        lines.append("")
        lines.append("| Budget | Pairs | W/T/L | Mean saving | Median saving |")
        lines.append("|---:|---:|---:|---:|---:|")
        for result in comparison["budgets"]:
            mean_saving = result["mean_relative_saving_pct"]
            median_saving = result["median_relative_saving_pct"]
            lines.append(
                f"| {result['budget_s']:g} s | {result['paired_units']} | "
                f"{result['wins']}/{result['ties']}/{result['losses']} | "
                f"{f'{mean_saving:.3f}%' if mean_saving is not None else '--'} | "
                f"{f'{median_saving:.3f}%' if median_saving is not None else '--'} |"
            )
        lines.append("")
    nonmonotone = diagnostics["nonmonotone_budget_pairs"]
    lines.extend(
        [
            "## Audit boundary",
            "",
            f"All {len(diagnostics['initial_consistency'])} method groups match on initial cost. "
            f"Fresh-process budget curves contain {len(nonmonotone)} cost regressions when moving to a longer budget; these are retained rather than repaired post hoc.",
            "Roots from one source are correlated, and one timing repeat is not a variance estimate. Win counts are descriptive paired outcomes.",
            f"{len(diagnostics['failed_worker_attempts'])} failed worker attempts are retained in the diagnostics. Per-method wall times describe the returned attempt; failed attempts and configured cooldowns are additional execution overhead.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("work/research_2026_10_08/budget_curve_summary"),
    )
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    configuration = payload["metadata"].get(
        "configuration", payload["metadata"].get("config")
    )
    expected = (
        sum(len(dataset["roots"]) for dataset in configuration["datasets"])
        * len(configuration["methods"])
        * len(configuration.get("budgets_s", [configuration.get("budget_s")]))
        * configuration.get("repeats", 1)
    )
    if len(payload["records"]) != expected:
        raise ValueError(f"expected {expected} records, found {len(payload['records'])}")
    hash_checks = audit_hashes(payload, Path.cwd())
    if not all(check["matches"] for check in hash_checks):
        changed = [check["path"] for check in hash_checks if not check["matches"]]
        raise ValueError(f"frozen hashes no longer match: {changed}")

    rows, summaries, comparisons, diagnostics = summarize_budget(payload)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(args.results),
                "hash_checks": hash_checks,
                "rows": rows,
                "summaries": summaries,
                "comparisons": comparisons,
                "diagnostics": diagnostics,
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
        _markdown(summaries, comparisons, diagnostics)
    )
    print(f"summarized {len(rows)} method-budget units")


if __name__ == "__main__":
    main()
