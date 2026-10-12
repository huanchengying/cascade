#!/usr/bin/env python3
"""Audit the applicability of same-pair dominance on the existing unit-duration trace.

With start=t, arrival=t+1 and weight=1, domination requires equal timestamps.
Thus only duplicate (source, destination, timestamp) events can be removed.
This is an applicability diagnostic, not a solver performance benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("sx-stackoverflow"))
    parser.add_argument("--prefixes", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--output", type=Path, default=Path("audit_stackoverflow_dominance_20260911.json"))
    args = parser.parse_args()
    if not args.prefixes or min(args.prefixes) < 1:
        parser.error("prefixes must be positive")
    checkpoints = set(args.prefixes)
    unique = set()
    count = 0
    previous_time = None
    first_inversion = None
    checksum = hashlib.sha256()
    records = []
    with args.input.open("rb") as stream:
        for raw in stream:
            if not raw.strip() or raw.lstrip().startswith(b"#"):
                continue
            fields = raw.split()
            if len(fields) != 3:
                raise ValueError("expected exactly source destination timestamp")
            event = tuple(map(int, fields))
            count += 1
            checksum.update(raw)
            unique.add(event)
            if previous_time is not None and event[2] < previous_time and first_inversion is None:
                first_inversion = {"event": count, "previous_time": previous_time, "current_time": event[2]}
            previous_time = event[2]
            if count in checkpoints:
                records.append({
                    "input_events": count, "unique_events": len(unique),
                    "removable_events": count - len(unique),
                    "removable_fraction": (count - len(unique)) / count,
                    "first_timestamp_inversion": first_inversion,
                    "prefix_sha256": checksum.hexdigest(),
                })
            if count == max(checkpoints):
                break
    payload = {
        "input": str(args.input.resolve()),
        "model": "directed events, duration=1, weight=1; same-pair dominance equals exact duplicate removal",
        "scope": "prefix diagnostic; not full-trace data and not weighted-MST solution quality",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
