#!/usr/bin/env python3
"""Freeze official SNAP communication histories for structural experiments."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

from prepare_root_selection_20261008 import select_roots
from temporal_mst_prototype import TemporalGraph


DIRECTORY = Path("work/research_2026_10_08")
DATASETS = (
    ("college-msg", "CollegeMsg.html", "CollegeMsg.txt.gz", 1899, 59835),
    (
        "email-eu-core", "email-Eu-core-temporal.html",
        "email-Eu-core-temporal.txt.gz", 986, 332334,
    ),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_dataset(specification, directory):
    name, page_name, filename, expected_vertices, expected_events = specification
    source_url = "https://snap.stanford.edu/data/" + filename
    raw_path = Path("work/data/raw") / filename
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.is_file():
        temporary = raw_path.with_suffix(raw_path.suffix + ".part")
        with urllib.request.urlopen(source_url, timeout=45) as response:
            temporary.write_bytes(response.read())
        temporary.replace(raw_path)
    rows = []
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for row_number, line in enumerate(stream):
            if line.startswith("#") or not line.strip():
                continue
            source, target, timestamp = line.split()
            rows.append((source, target, int(timestamp), row_number))
    vertices = sorted({value for row in rows for value in row[:2]}, key=int)
    if len(vertices) != expected_vertices or len(rows) != expected_events:
        raise ValueError(f"official SNAP statistics disagree for {name}")
    origin = min(row[2] for row in rows)
    rows.sort(key=lambda row: (row[2], row[3]))
    metadata = {
        "dataset": name,
        "source_page": "https://snap.stanford.edu/data/" + page_name,
        "source_url": source_url,
        "raw_sha256": sha256(raw_path),
        "raw_rows_used": len(rows),
        "time_model": "recorded timestamp minus minimum; synthetic one-second duration",
        "duration": 1,
        "weight_model": "unit cost; the source records no event weight",
        "sampling": "complete directed temporal history; no sampling",
        "time_span_seconds": max(row[2] for row in rows) - origin,
    }
    graph_data = {
        "metadata": metadata,
        "vertices": vertices,
        "edges": [
            {
                "id": f"{name}:{row_number}", "u": source, "v": target,
                "start": timestamp - origin, "arrival": timestamp - origin + 1,
                "weight": 1.0,
            }
            for source, target, timestamp, row_number in rows
        ],
    }
    input_path = directory / "snap_inputs" / f"{name}.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(graph_data, separators=(",", ":")) + "\n")
    selected, ranking = select_roots(
        TemporalGraph.from_json(input_path), candidate_count=20, selection_count=2,
    )
    return {
        "name": name, "input": input_path.as_posix(),
        "roots": [row["root"] for row in selected],
        "scale_events": len(rows),
    }, {
        **metadata, "input": input_path.as_posix(),
        "input_sha256": sha256(input_path), "vertices": len(vertices),
        "events": len(rows), "selected_roots": selected,
        "candidate_ranking": ranking,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    prepared = [prepare_dataset(specification, args.directory) for specification in DATASETS]
    configuration = {
        "datasets": [entry for entry, _ in prepared],
        "methods": ["path_union", "ea_branching", "star", "cascade"],
        "repeats": 1, "hard_timeout_s": 180,
    }
    provenance = {
        "generator": Path(__file__).name, "generator_sha256": sha256(Path(__file__)),
        "root_rule": "top 20 event outdegrees, then temporal reachability/outdegree/lexical ID; top two roots",
        "purpose": "structural feasibility, end-to-end time and process RSS; unit weights give no weighted-quality comparison",
        "datasets": [entry for _, entry in prepared],
    }
    for filename, payload in (
        ("snap_structural_config.json", configuration),
        ("snap_provenance_20261008.json", provenance),
    ):
        (args.directory / filename).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(configuration, indent=2))


if __name__ == "__main__":
    main()
