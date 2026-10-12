#!/usr/bin/env python3
"""Convert official NYC TLC Yellow Taxi records into temporal graphs."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow
import pyarrow.parquet as parquet


DEFAULT_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2025-01.parquet"
)
SOURCE_PAGE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
DATASET = "nyc-tlc-yellow-2025-01"
MONTH_START = datetime(2025, 1, 1)
MONTH_END = datetime(2025, 2, 1)
EPOCH = datetime(1970, 1, 1)
WEIGHTS = {
    "duration": ("duration_s", "recorded trip duration in seconds"),
    "distance": ("trip_distance_miles", "recorded trip distance in miles"),
    "total-amount": ("total_amount_usd", "recorded total amount in US dollars"),
}
COLUMNS = (
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "total_amount",
)


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


def _seconds(timestamp: datetime) -> float:
    return (timestamp - EPOCH).total_seconds()


def _summary(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round(fraction * (len(ordered) - 1))]

    return {
        "min": ordered[0],
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _valid_record(columns: Dict[str, list], offset: int) -> Tuple[tuple, str]:
    pickup = columns["tpep_pickup_datetime"][offset]
    dropoff = columns["tpep_dropoff_datetime"][offset]
    source = columns["PULocationID"][offset]
    target = columns["DOLocationID"][offset]
    distance = columns["trip_distance"][offset]
    total_amount = columns["total_amount"][offset]
    if pickup is None or not MONTH_START <= pickup < MONTH_END:
        return (), "pickup_outside_month"
    if dropoff is None or dropoff <= pickup:
        return (), "nonpositive_duration"
    duration = (dropoff - pickup).total_seconds()
    if duration > 86_400:
        return (), "duration_over_24h"
    if (
        source is None
        or target is None
        or not 1 <= source <= 263
        or not 1 <= target <= 263
    ):
        return (), "invalid_taxi_zone"
    if source == target:
        return (), "same_taxi_zone"
    if distance is None or not math.isfinite(distance) or distance <= 0:
        return (), "invalid_distance"
    if total_amount is None or not math.isfinite(total_amount) or total_amount <= 0:
        return (), "invalid_total_amount"
    return (
        str(source),
        str(target),
        _seconds(pickup),
        _seconds(dropoff),
        float(duration),
        float(distance),
        float(total_amount),
    ), ""


def convert_tlc_taxi(
    input_path: Path,
    output_dir: Path,
    *,
    sample_size: int,
    batch_size: int = 65_536,
) -> Dict[str, Path]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    source = parquet.ParquetFile(input_path)
    missing = sorted(set(COLUMNS) - set(source.schema_arrow.names))
    if missing:
        raise ValueError(f"missing TLC columns: {missing}")

    sample: List[Tuple[int, int, tuple]] = []
    rejected: Counter[str] = Counter()
    valid_rows = 0
    raw_row = 0
    for batch in source.iter_batches(batch_size=batch_size, columns=COLUMNS):
        columns = batch.to_pydict()
        for offset in range(batch.num_rows):
            record, reason = _valid_record(columns, offset)
            if reason:
                rejected[reason] += 1
                raw_row += 1
                continue
            rank = int.from_bytes(
                hashlib.blake2b(
                    f"{DATASET}:{raw_row}".encode("ascii"), digest_size=8
                ).digest(),
                "big",
            )
            full_record = (*record, raw_row)
            item = (-rank, -raw_row, full_record)
            if len(sample) < sample_size:
                heapq.heappush(sample, item)
            elif item[:2] > sample[0][:2]:
                heapq.heapreplace(sample, item)
            valid_rows += 1
            raw_row += 1

    if not sample:
        raise ValueError("no valid TLC trips remain after filtering")
    records = [item[2] for item in sample]
    records.sort(key=lambda record: (record[2], record[3], record[7]))
    origin = min(record[2] for record in records)
    vertices = sorted(
        {record[0] for record in records} | {record[1] for record in records},
        key=int,
    )
    base_edges = [
        {
            "id": f"{DATASET}:{record[7]}",
            "u": record[0],
            "v": record[1],
            "start": record[2] - origin,
            "arrival": record[3] - origin,
            "duration_s": record[4],
            "trip_distance_miles": record[5],
            "total_amount_usd": record[6],
            "raw_row": record[7],
        }
        for record in records
    ]
    common_metadata = {
        "dataset": DATASET,
        "source_page": SOURCE_PAGE,
        "source_url": DEFAULT_URL,
        "raw_sha256": _sha256(input_path),
        "parquet_rows": source.metadata.num_rows,
        "valid_rows_scanned": valid_rows,
        "invalid_rows_filtered": sum(rejected.values()),
        "rejection_counts": dict(sorted(rejected.items())),
        "sample_size": len(records),
        "sampling": (
            "smallest unsigned 64-bit BLAKE2b hashes of dataset name and "
            "zero-based Parquet row number"
        ),
        "time_model": (
            "recorded tpep_pickup_datetime and tpep_dropoff_datetime, shifted "
            "by the minimum sampled pickup"
        ),
        "filters": [
            "pickup in January 2025",
            "positive recorded duration at most 24 hours",
            "distinct official taxi zones in [1,263]",
            "positive finite recorded trip_distance",
            "positive finite recorded total_amount",
        ],
        "pyarrow": pyarrow.__version__,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for name, (field, description) in WEIGHTS.items():
        edges = [{**edge, "weight": edge[field]} for edge in base_edges]
        payload = {
            "metadata": {
                **common_metadata,
                "weight_model": description,
                "weight_summary": _summary([edge[field] for edge in base_edges]),
            },
            "vertices": vertices,
            "edges": edges,
        }
        path = output_dir / f"{DATASET}-{name}-{sample_size // 1000}k.json"
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        outputs[name] = path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("work/data/raw/yellow_tripdata_2025-01.parquet"),
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("work/data/processed")
    )
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()
    _download(args.url, args.input)
    outputs = convert_tlc_taxi(
        args.input,
        args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
