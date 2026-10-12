#!/usr/bin/env python3
"""Download and convert real weighted temporal graphs used by the MST study."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DATASETS: Dict[str, Dict[str, str]] = {
    "bitcoin-alpha": {
        "url": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
        "page": "https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html",
        "filename": "soc-sign-bitcoinalpha.csv.gz",
    },
    "bitcoin-otc": {
        "url": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
        "page": "https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html",
        "filename": "soc-sign-bitcoinotc.csv.gz",
    },
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bitcoin_rows(
    path: Path,
    max_events: Optional[int],
) -> List[Tuple[str, str, int, float, int]]:
    rows: List[Tuple[str, str, int, float, int]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        for row_number, row in enumerate(reader):
            if max_events is not None and len(rows) >= max_events:
                break
            if len(row) != 4:
                raise ValueError(f"line {row_number + 1}: expected four columns")
            source, target, rating, timestamp = row
            rows.append((source, target, int(rating), float(timestamp), row_number))
    rows.sort(key=lambda row: (row[3], row[4]))
    return rows


def convert_bitcoin_dataset(
    name: str,
    raw_dir: Path,
    output_dir: Path,
    *,
    max_events: Optional[int] = None,
    duration: float = 1.0,
) -> Path:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}")
    if duration <= 0:
        raise ValueError("duration must be positive")
    spec = DATASETS[name]
    raw_path = raw_dir / spec["filename"]
    _download(spec["url"], raw_path)
    rows = _load_bitcoin_rows(raw_path, max_events)
    if not rows:
        raise ValueError(f"dataset {name} is empty")
    origin = min(row[3] for row in rows)
    vertices = sorted(
        {row[0] for row in rows} | {row[1] for row in rows},
        key=lambda value: (int(value) if value.lstrip("-").isdigit() else value),
    )
    edges = []
    rating_histogram: Dict[str, int] = {}
    for position, (source, target, rating, timestamp, raw_row) in enumerate(rows):
        rating_histogram[str(rating)] = rating_histogram.get(str(rating), 0) + 1
        start = timestamp - origin
        edges.append(
            {
                "id": f"{name}:{raw_row}",
                "u": source,
                "v": target,
                "start": start,
                "arrival": start + duration,
                "weight": float(11 - rating),
                "rating": rating,
                "raw_timestamp": timestamp,
            }
        )
    payload = {
        "metadata": {
            "dataset": name,
            "source_page": spec["page"],
            "source_url": spec["url"],
            "raw_sha256": _sha256(raw_path),
            "raw_rows_used": len(rows),
            "time_model": "start = timestamp - minimum timestamp; arrival = start + duration",
            "duration": duration,
            "weight_model": "trust_penalty = 11 - signed_rating",
            "weight_range": [1, 21],
            "rating_histogram": rating_histogram,
        },
        "vertices": vertices,
        "edges": edges,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{max_events}" if max_events is not None else ""
    output_path = output_dir / f"{name}{suffix}.json"
    output_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("work/data/raw"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("work/data/processed")
    )
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()
    names: Iterable[str] = DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        output = convert_bitcoin_dataset(
            name,
            args.raw_dir,
            args.output_dir,
            max_events=args.max_events,
            duration=args.duration,
        )
        print(output)


if __name__ == "__main__":
    main()
