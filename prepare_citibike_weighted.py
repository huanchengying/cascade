#!/usr/bin/env python3
"""Convert official Citi Bike trips into a real weighted temporal graph.

The converter streams the monthly ZIP archive, keeps a deterministic min-hash
sample of valid rides, uses the recorded ride start/end as temporal edge times,
and assigns the great-circle station distance as the physical edge weight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import math
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_URL = "https://s3.amazonaws.com/tripdata/202604-citibike-tripdata.zip"
SOURCE_PAGE = "https://citibikenyc.com/system-data"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and zipfile.is_zipfile(destination):
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


def _seconds(timestamp: str) -> float:
    parsed = datetime.fromisoformat(timestamp)
    return (parsed - datetime(1970, 1, 1)).total_seconds()


def _haversine_meters(
    start_latitude: float,
    start_longitude: float,
    end_latitude: float,
    end_longitude: float,
) -> float:
    earth_radius = 6_371_008.8
    phi_one = math.radians(start_latitude)
    phi_two = math.radians(end_latitude)
    delta_phi = phi_two - phi_one
    delta_lambda = math.radians(end_longitude - start_longitude)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_one) * math.cos(phi_two) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * earth_radius * math.asin(min(1.0, math.sqrt(value)))


def convert_citibike(
    input_path: Path,
    output_path: Path,
    *,
    sample_size: int,
    max_duration_s: float = 86_400.0,
) -> Path:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if max_duration_s <= 0:
        raise ValueError("max_duration_s must be positive")
    if not zipfile.is_zipfile(input_path):
        raise ValueError(f"not a valid ZIP archive: {input_path}")

    sample: List[Tuple[int, int, tuple]] = []
    valid_rows = 0
    invalid_rows = 0
    station_names: Dict[str, str] = {}
    with zipfile.ZipFile(input_path) as archive:
        members = sorted(
            member for member in archive.namelist() if member.lower().endswith(".csv")
        )
        for member in members:
            with archive.open(member) as raw_stream:
                stream = io.TextIOWrapper(raw_stream, encoding="utf-8-sig", newline="")
                for row_number, row in enumerate(csv.DictReader(stream), start=2):
                    try:
                        ride_id = row["ride_id"].strip()
                        source = row["start_station_id"].strip()
                        target = row["end_station_id"].strip()
                        source_name = row["start_station_name"].strip()
                        target_name = row["end_station_name"].strip()
                        if not ride_id or not source or not target or source == target:
                            raise ValueError
                        start = _seconds(row["started_at"])
                        arrival = _seconds(row["ended_at"])
                        duration = arrival - start
                        if not 0 < duration <= max_duration_s:
                            raise ValueError
                        coordinates = tuple(
                            float(row[key])
                            for key in ("start_lat", "start_lng", "end_lat", "end_lng")
                        )
                        if not all(math.isfinite(value) for value in coordinates):
                            raise ValueError
                        distance = _haversine_meters(*coordinates)
                        if not math.isfinite(distance) or distance <= 0:
                            raise ValueError
                    except (KeyError, TypeError, ValueError):
                        invalid_rows += 1
                        continue
                    valid_rows += 1
                    station_names.setdefault(source, source_name)
                    station_names.setdefault(target, target_name)
                    rank = int.from_bytes(
                        hashlib.blake2b(ride_id.encode("utf-8"), digest_size=8).digest(),
                        "big",
                    )
                    record = (
                        ride_id,
                        source,
                        target,
                        start,
                        arrival,
                        distance,
                        duration,
                        member,
                        row_number,
                    )
                    item = (-rank, valid_rows, record)
                    if len(sample) < sample_size:
                        heapq.heappush(sample, item)
                    elif item[0] > sample[0][0]:
                        heapq.heapreplace(sample, item)

    records = [item[2] for item in sample]
    records.sort(key=lambda record: (record[3], record[4], record[0]))
    origin = min(record[3] for record in records)
    vertices = sorted(
        {record[1] for record in records} | {record[2] for record in records}
    )
    edges = [
        {
            "id": record[0],
            "u": record[1],
            "v": record[2],
            "start": record[3] - origin,
            "arrival": record[4] - origin,
            "weight": record[5],
            "duration_s": record[6],
        }
        for record in records
    ]
    payload = {
        "metadata": {
            "dataset": "citibike-202604",
            "source_page": SOURCE_PAGE,
            "source_url": DEFAULT_URL,
            "raw_sha256": _sha256(input_path),
            "valid_rows_scanned": valid_rows,
            "invalid_rows_filtered": invalid_rows,
            "sample_size": len(records),
            "sampling": "smallest unsigned 64-bit BLAKE2b hashes of ride_id",
            "time_model": "recorded started_at and ended_at, shifted by minimum start",
            "weight_model": "haversine distance in meters between recorded station coordinates",
            "max_duration_s": max_duration_s,
            "archive_members": sorted(
                {record[7] for record in records}
            ),
            "station_names": {vertex: station_names.get(vertex, "") for vertex in vertices},
        },
        "vertices": vertices,
        "edges": edges,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("work/data/raw/202604-citibike-tripdata.zip"),
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/data/processed/citibike-202604-distance-100k.json"),
    )
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--max-duration-s", type=float, default=86_400.0)
    args = parser.parse_args()
    _download(args.url, args.input)
    print(
        convert_citibike(
            args.input,
            args.output,
            sample_size=args.sample_size,
            max_duration_s=args.max_duration_s,
        )
    )


if __name__ == "__main__":
    main()
