#!/usr/bin/env python3
"""Summarize complete unit-weight SNAP runs without weighted-quality claims."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from verify_artifact_20261008 import audit_results, audit_snap_provenance


DIRECTORY = Path("work/research_2026_10_08")


def main():
    path = DIRECTORY / "snap_structural_results.json"
    audit = audit_results(path.resolve(), Path.cwd())
    provenance_audit = audit_snap_provenance(Path.cwd())
    payload = json.loads(path.read_text())
    rows = []
    for dataset in payload["metadata"]["config"]["datasets"]:
        for method in payload["metadata"]["config"]["methods"]:
            records = [
                record for record in payload["records"]
                if record["dataset"] == dataset["name"] and record["method"] == method
            ]
            for record in records:
                if not record.get("validated"):
                    raise ValueError("SNAP method returned no valid temporal tree")
                optimum = record["reachable_count"] - 1
                if record["selected_events"] != optimum or record["cost"] != optimum:
                    raise ValueError("unit-cost tree disagrees with the analytic optimum")
            rows.append({
                "dataset": dataset["name"], "method": method,
                "roots": len(records), "validated": len(records),
                "median_wall_s": statistics.median(record["wall_time_s"] for record in records),
                "median_rss_mib": statistics.median(record["peak_rss_mib"] for record in records),
                "min_reachable": min(record["reachable_count"] for record in records),
                "max_reachable": max(record["reachable_count"] for record in records),
                "all_unit_cost_optimal": True,
            })
    summary = {
        "audit": audit, "provenance_audit": provenance_audit, "rows": rows,
        "scope": "full histories, two correlated roots, one process per method-root; no timing variance estimate",
        "optimality_reason": "every feasible tree has one selected incoming event per reachable non-root vertex and every event costs one",
    }
    (DIRECTORY / "snap_structural_summary_20261008.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (DIRECTORY / "snap_structural_summary_20261008.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cascade = {row["dataset"]: row for row in rows if row["method"] == "cascade"}
    earliest = {row["dataset"]: row for row in rows if row["method"] == "ea_branching"}
    college, email = cascade["college-msg"], cascade["email-eu-core"]
    findings = (
        r"\par\noindent \textbf{Full SNAP histories.} "
        "Sixteen fresh-process runs compare paths, EA, star, and Cascade on "
        "two roots in each complete communication history. CollegeMsg roots "
        "reach 1,773--1,776 vertices; email-Eu-core roots reach 965. "
        f"Cascade's median end-to-end times are {college['median_wall_s']:.3f} and "
        f"{email['median_wall_s']:.3f} seconds, with {college['median_rss_mib']:.1f} and "
        f"{email['median_rss_mib']:.1f} MiB RSS. Every method returns a valid tree "
        r"of cost $|\Rset|-1$, the analytic unit-cost optimum. "
        f"EA is faster at {earliest['college-msg']['median_wall_s']:.3f} and "
        f"{earliest['email-eu-core']['median_wall_s']:.3f} seconds: batch search adds "
        "work when costs cannot improve. These full-history measurements "
        "extend structural coverage to 332,334 edges; assigned unit weights "
        "do not supply evidence of weighted-quality improvement.\n"
    )
    (DIRECTORY / "snap_structural_findings.tex").write_text(findings)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
