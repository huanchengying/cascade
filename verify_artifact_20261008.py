#!/usr/bin/env python3
"""Verify frozen experiment coverage, hashes, bounds, and portable provenance."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import unittest
from pathlib import Path

from analyze_cascade_results import audit_hashes
from prepare_root_selection_20261008 import select_roots
from temporal_mst_prototype import TemporalGraph


RESULT_FILES = (
    "submission_exact_quality_20261008.json",
    "submission_speedsteiner_exact_20261008.json",
    "cascade_multiroot_results.json",
    "cascade_convergence_results.json",
    "budget_curve_results.json",
    "speedsteiner_100k_results.json",
    "microprofile_results.json",
    "sensitivity_results.json",
    "additional_exact_results.json",
    "snap_structural_results.json",
)
RESEARCH_DIRECTORY = Path("work/research_2026_10_08")
LOCAL_IDENTITY = re.compile(r"(?<![\w/])/(?:Users|home)/", re.IGNORECASE)


def audit_results(path: Path, base_dir: Path) -> dict:
    payload = json.loads(path.read_text())
    configuration = payload["metadata"].get(
        "configuration", payload["metadata"].get("config")
    )
    budgets = configuration.get("budgets_s", [configuration.get("budget_s")])
    expected_keys = {
        (entry["name"], str(root), method, budget, repeat)
        for entry in configuration["datasets"]
        for root in entry["roots"]
        for method in configuration["methods"]
        for budget in budgets
        for repeat in range(configuration.get("repeats", 1))
    }
    actual_keys = [
        (
            record["dataset"],
            str(record["root"]),
            record["method"],
            record.get("budget_s"),
            record["repeat"],
        )
        for record in payload["records"]
    ]
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError(f"duplicate records in {path}")
    if set(actual_keys) != expected_keys:
        raise ValueError(f"incomplete or mismatched experiment matrix in {path}")
    checks = audit_hashes(payload, base_dir)
    if not checks or not all(check["matches"] for check in checks):
        changed = [check["path"] for check in checks if not check["matches"]]
        raise ValueError(f"source/input hash mismatch in {path}: {changed}")
    if LOCAL_IDENTITY.search(path.read_text()):
        raise ValueError(f"local identity in result metadata: {path}")
    statuses = {}
    for record in payload["records"]:
        status = record.get("status", "feasible")
        statuses[status] = statuses.get(status, 0) + 1
        if record.get("validated") and record.get("lower_bound", 0) > record["cost"] + 1e-7 * max(1.0, record["cost"]):
            raise ValueError(f"global bound exceeds tree cost in {path}")
    return {
        "result_file": path.relative_to(base_dir).as_posix(),
        "expected_records": len(expected_keys),
        "records": len(actual_keys),
        "hashes_match": True,
        "statuses": statuses,
        "failed_worker_attempts": sum(
            len(record.get("worker_failures", [])) for record in payload["records"]
        ),
    }


def audit_root_selection(base_dir: Path) -> dict:
    path = base_dir / RESEARCH_DIRECTORY / "root_selection_20261008.json"
    payload = json.loads(path.read_text())
    for entry in payload["datasets"]:
        actual = hashlib.sha256((base_dir / entry["input"]).read_bytes()).hexdigest()
        if actual != entry["input_sha256"]:
            raise ValueError(f"root selection input changed: {entry['input']}")
        ranking = sorted(
            entry["candidate_ranking"],
            key=lambda row: (
                -row["temporal_reachability"],
                -row["event_outdegree"],
                row["root"],
            ),
        )
        selected = [row["root"] for row in ranking[:payload["rule"]["selection_count"]]]
        if selected != entry["selected_roots"]:
            raise ValueError("root-selection ranking mismatch")
        graph = TemporalGraph.from_json(base_dir / entry["input"])
        recomputed, candidates = select_roots(
            graph,
            candidate_count=payload["rule"]["candidate_count"],
            selection_count=payload["rule"]["selection_count"],
        )
        if candidates != entry["candidate_ranking"] or selected != [row["root"] for row in recomputed]:
            raise ValueError("root-selection reachability/outdegree evidence mismatch")
    generator = base_dir / payload["generator"]
    if hashlib.sha256(generator.read_bytes()).hexdigest() != payload["generator_sha256"]:
        raise ValueError("root-selection generator changed")
    return {"rule_verified": True, "datasets": len(payload["datasets"])}


def audit_snap_provenance(base_dir: Path) -> dict:
    path = base_dir / RESEARCH_DIRECTORY / "snap_provenance_20261008.json"
    payload = json.loads(path.read_text())
    generator = base_dir / payload["generator"]
    if hashlib.sha256(generator.read_bytes()).hexdigest() != payload["generator_sha256"]:
        raise ValueError("SNAP preparation source changed")
    for entry in payload["datasets"]:
        input_path = base_dir / entry["input"]
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != entry["input_sha256"]:
            raise ValueError("SNAP input changed")
        graph = TemporalGraph.from_json(input_path)
        if len(graph.vertices) != entry["vertices"] or len(graph.edges) != entry["events"]:
            raise ValueError("SNAP graph statistics changed")
        if any(event.weight != 1 or event.arrival - event.start != 1 for event in graph.edges):
            raise ValueError("SNAP unit-weight or synthetic-duration semantics changed")
        selected, ranking = select_roots(graph, candidate_count=20, selection_count=2)
        if selected != entry["selected_roots"] or ranking != entry["candidate_ranking"]:
            raise ValueError("SNAP structure-selected roots changed")
    return {"datasets": len(payload["datasets"]), "complete_histories": True, "roots_verified": True}


def verify_manifest(base_dir: Path) -> dict | None:
    path = base_dir / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    for name, expected in manifest["sha256"].items():
        bundled = base_dir / name
        if not bundled.is_file() or hashlib.sha256(bundled.read_bytes()).hexdigest() != expected:
            raise ValueError(f"artifact manifest mismatch: {name}")
    return {"files": len(manifest["sha256"]), "all_match": True}


def run_tests() -> dict:
    suite = unittest.defaultTestLoader.discover(".")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = unittest.TextTestRunner(stream=output).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(output.getvalue())
    return {"run": result.testsRun, "failures": 0, "errors": 0, "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base_dir = Path.cwd()
    audits = [
        audit_results(base_dir / RESEARCH_DIRECTORY / name, base_dir)
        for name in RESULT_FILES
    ]
    payload = {
        "experiments": audits,
        "total_records": sum(audit["records"] for audit in audits),
        "root_selection": audit_root_selection(base_dir),
        "snap_provenance": audit_snap_provenance(base_dir),
        "manifest": verify_manifest(base_dir),
    }
    if args.run_tests:
        payload["tests"] = run_tests()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
