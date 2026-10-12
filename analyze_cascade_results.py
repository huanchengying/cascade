#!/usr/bin/env python3
"""Audit and pair conservative-star and cascade-star benchmark records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PAIR_FIELDS = (
    "dataset",
    "root",
    "baseline_cost",
    "treatment_cost",
    "absolute_saving",
    "relative_saving_pct",
    "outcome",
    "baseline_wall_s",
    "treatment_wall_s",
    "wall_overhead_s",
    "wall_ratio",
    "cascade_moves",
    "activated_children",
    "relaxed_deadline_vertices",
    "baseline_rounds",
    "baseline_moves",
    "baseline_converged",
)


def _median(records: Iterable[dict], key: str, default: float = 0.0) -> float:
    values = [float(record[key]) for record in records if key in record]
    return statistics.median(values) if values else default


def _configuration(payload: dict) -> dict:
    metadata = payload["metadata"]
    return metadata.get("configuration", metadata.get("config", {}))


def audit_hashes(payload: dict, base_dir: Path) -> list[dict]:
    checks = []
    metadata = payload["metadata"]
    for category, entries in (
        ("source", metadata.get("sha256", {})),
        ("input", metadata.get("input_sha256", {})),
    ):
        for name, expected in entries.items():
            path = Path(name)
            if not path.is_absolute():
                path = base_dir / path
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            checks.append(
                {
                    "category": category,
                    "path": name,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "matches": actual == expected,
                }
            )
    return checks


def _aggregate(label: str, pairs: list[dict]) -> dict:
    ratios = [pair["treatment_cost"] / pair["baseline_cost"] for pair in pairs]
    positive_ratios = [ratio for ratio in ratios if ratio > 0]
    return {
        "dataset": label,
        "paired_units": len(pairs),
        "wins": sum(pair["outcome"] == "win" for pair in pairs),
        "ties": sum(pair["outcome"] == "tie" for pair in pairs),
        "losses": sum(pair["outcome"] == "loss" for pair in pairs),
        "total_absolute_saving": math.fsum(pair["absolute_saving"] for pair in pairs),
        "mean_relative_saving_pct": statistics.fmean(
            pair["relative_saving_pct"] for pair in pairs
        ),
        "median_relative_saving_pct": statistics.median(
            pair["relative_saving_pct"] for pair in pairs
        ),
        "geometric_mean_cost_ratio": (
            math.exp(statistics.fmean(math.log(ratio) for ratio in positive_ratios))
            if positive_ratios
            else None
        ),
        "median_wall_overhead_s": statistics.median(
            pair["wall_overhead_s"] for pair in pairs
        ),
        "median_wall_ratio": statistics.median(pair["wall_ratio"] for pair in pairs),
        "wins_with_relaxed_deadlines": sum(
            pair["outcome"] == "win" and pair["relaxed_deadline_vertices"] > 0
            for pair in pairs
        ),
        "verified_converged_baselines": sum(
            pair["baseline_converged"] is True for pair in pairs
        ),
        "capped_baselines": sum(
            pair["baseline_converged"] is False for pair in pairs
        ),
        "wins_after_verified_convergence": sum(
            pair["baseline_converged"] is True and pair["outcome"] == "win"
            for pair in pairs
        ),
    }


def summarize_payload(
    payload: dict,
    *,
    baseline: str = "star",
    treatment: str = "cascade",
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-9,
) -> tuple[list[dict], list[dict]]:
    records = payload["records"]
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        if not record.get("validated", False):
            raise ValueError(
                f"unvalidated record: {record.get('dataset')} "
                f"root={record.get('root')} method={record.get('method')}"
            )
        groups[(record["dataset"], str(record["root"]), record["method"])].append(
            record
        )

    units = sorted({(dataset, root) for dataset, root, _ in groups})
    pairs = []
    for dataset, root in units:
        baseline_rows = groups.get((dataset, root, baseline), [])
        treatment_rows = groups.get((dataset, root, treatment), [])
        if not baseline_rows or not treatment_rows:
            raise ValueError(f"missing pair for {dataset} root={root}")
        if len(baseline_rows) != len(treatment_rows):
            raise ValueError(f"unequal repeat counts for {dataset} root={root}")

        baseline_cost = _median(baseline_rows, "cost")
        treatment_cost = _median(treatment_rows, "cost")
        saving = baseline_cost - treatment_cost
        tolerance = max(
            absolute_tolerance,
            relative_tolerance * max(1.0, abs(baseline_cost), abs(treatment_cost)),
        )
        outcome = "win" if saving > tolerance else "loss" if saving < -tolerance else "tie"
        baseline_wall = _median(baseline_rows, "wall_time_s")
        treatment_wall = _median(treatment_rows, "wall_time_s")
        convergence_known = all(
            "rounds" in record and "accepted_moves" in record
            for record in baseline_rows
        )
        baseline_converged = (
            all(record["rounds"] > record["accepted_moves"] for record in baseline_rows)
            if convergence_known
            else None
        )
        pair = {
            "dataset": dataset,
            "root": root,
            "repeats_per_method": len(baseline_rows),
            "baseline_method": baseline,
            "treatment_method": treatment,
            "baseline_cost": baseline_cost,
            "treatment_cost": treatment_cost,
            "absolute_saving": saving,
            "relative_saving_pct": 100.0 * saving / baseline_cost if baseline_cost else 0.0,
            "outcome": outcome,
            "baseline_wall_s": baseline_wall,
            "treatment_wall_s": treatment_wall,
            "wall_overhead_s": treatment_wall - baseline_wall,
            "wall_ratio": treatment_wall / baseline_wall if baseline_wall else math.inf,
            "cascade_moves": _median(treatment_rows, "cascade_moves"),
            "activated_children": _median(treatment_rows, "activated_children"),
            "relaxed_deadline_vertices": _median(
                treatment_rows, "relaxed_deadline_vertices"
            ),
            "baseline_rounds": _median(baseline_rows, "rounds"),
            "baseline_moves": _median(baseline_rows, "accepted_moves"),
            "baseline_converged": baseline_converged,
        }
        pairs.append(pair)

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_dataset[pair["dataset"]].append(pair)
    summaries = [_aggregate(dataset, rows) for dataset, rows in sorted(by_dataset.items())]
    summaries.append(_aggregate("ALL", pairs))
    return pairs, summaries


def _markdown(pairs: list[dict], summaries: list[dict], hashes_match: bool) -> str:
    overall = summaries[-1]
    lines = [
        "# Paired cascade-star audit",
        "",
        f"Frozen source and input hashes match: **{'yes' if hashes_match else 'no'}**.",
        "The statistical unit is one dataset-root pair; timing repeats are collapsed by median.",
        "Roots from the same source are correlated, so win counts are descriptive rather than an independence-based significance test.",
        "",
        "## Overall",
        "",
        f"Cascade records {overall['wins']} wins, {overall['ties']} ties, and {overall['losses']} losses across {overall['paired_units']} paired units.",
        f"The median relative cost change is {overall['median_relative_saving_pct']:.4f}% and the geometric-mean cost ratio is {overall['geometric_mean_cost_ratio']:.6f}.",
        f"Median additional wall time is {overall['median_wall_overhead_s']:.3f} s ({overall['median_wall_ratio']:.3f}x total runtime).",
        f"{overall['wins_with_relaxed_deadlines']} strict wins explicitly relax at least one incumbent departure deadline.",
        f"Only {overall['verified_converged_baselines']} baselines have a recorded no-improvement round; {overall['wins_after_verified_convergence']} of those are strict cascade wins. The other {overall['capped_baselines']} baselines exhaust their round cap and are not certified local optima.",
        "",
        "## Dataset summary",
        "",
        "| Dataset | Pairs | W/T/L | Mean save | Median save | Cost ratio | Median time ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries[:-1]:
        lines.append(
            f"| {summary['dataset']} | {summary['paired_units']} | "
            f"{summary['wins']}/{summary['ties']}/{summary['losses']} | "
            f"{summary['mean_relative_saving_pct']:.4f}% | "
            f"{summary['median_relative_saving_pct']:.4f}% | "
            f"{summary['geometric_mean_cost_ratio']:.6f} | "
            f"{summary['median_wall_ratio']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## Paired results",
            "",
            "| Dataset | Root | Star | Cascade | Save | Outcome | Time ratio | Relaxed deadlines |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in pairs:
        lines.append(
            f"| {pair['dataset']} | {pair['root']} | {pair['baseline_cost']:.6f} | "
            f"{pair['treatment_cost']:.6f} | {pair['relative_saving_pct']:.4f}% | "
            f"{pair['outcome']} | {pair['wall_ratio']:.3f}x | "
            f"{pair['relaxed_deadline_vertices']:g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This experiment isolates additional search after the configured conservative-star round limit. Only a final no-improvement round establishes convergence. It does not compare equal wall-clock budgets: cascade spends extra time after the same incumbent. Budgeted end-to-end comparisons must be reported separately.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", default="star")
    parser.add_argument("--treatment", default="cascade")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("work/research_2026_10_08/cascade_multiroot_paired"),
    )
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    configuration = _configuration(payload)
    expected = (
        sum(len(dataset["roots"]) for dataset in configuration["datasets"])
        * len(configuration["methods"])
        * configuration.get("repeats", 1)
    )
    if len(payload["records"]) != expected:
        raise ValueError(f"expected {expected} records, found {len(payload['records'])}")
    hash_checks = audit_hashes(payload, Path.cwd())
    hashes_match = all(check["matches"] for check in hash_checks)
    if not hashes_match:
        changed = [check["path"] for check in hash_checks if not check["matches"]]
        raise ValueError(f"frozen hashes no longer match: {changed}")

    pairs, summaries = summarize_payload(
        payload, baseline=args.baseline, treatment=args.treatment
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(args.results),
                "baseline": args.baseline,
                "treatment": args.treatment,
                "hash_checks": hash_checks,
                "pairs": pairs,
                "summaries": summaries,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=PAIR_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)
    args.output_prefix.with_suffix(".md").write_text(
        _markdown(pairs, summaries, hashes_match)
    )
    overall = summaries[-1]
    print(
        f"paired {overall['paired_units']} units: "
        f"{overall['wins']} wins, {overall['ties']} ties, {overall['losses']} losses"
    )


if __name__ == "__main__":
    main()
