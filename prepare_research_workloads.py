#!/usr/bin/env python3
"""Explicit weight semantics and temporal windows for existing public inputs.

Trust penalties are transformations of observed ratings, not measured financial
costs. Citi Bike duration is a recorded physical measurement; geographic
distance is a derived straight-line proxy, not an observed route length.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Tuple


def transform_payload(payload: dict, policy: str, window: Optional[Tuple[float, float]] = None) -> dict:
    allowed = {"trust-linear", "trust-positive", "trust-exponential", "recorded-duration", "geographic-distance"}
    if policy not in allowed:
        raise ValueError(f"unknown weight policy {policy}")
    if window is not None and not (len(window) == 2 and all(math.isfinite(value) for value in window) and window[0] <= window[1]):
        raise ValueError("window must have finite ordered endpoints")
    edges = []
    skipped = 0
    for original in payload["edges"]:
        edge = dict(original)
        if window is not None and not (edge["start"] >= window[0] and edge["arrival"] <= window[1]):
            skipped += 1
            continue
        if policy.startswith("trust-"):
            if "rating" not in edge or not -10 <= edge["rating"] <= 10:
                raise ValueError("trust policies require a recorded rating in [-10,10]")
            if policy == "trust-positive" and edge["rating"] <= 0:
                skipped += 1
                continue
            edge["weight"] = math.exp(-edge["rating"] / 5) if policy == "trust-exponential" else 11.0 - edge["rating"]
        elif policy == "recorded-duration":
            duration = edge["arrival"] - edge["start"]
            if "duration_s" not in edge or not math.isclose(duration, edge["duration_s"], rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("recorded-duration requires consistent measured duration_s")
            edge["weight"] = edge["duration_s"]
        if not math.isfinite(edge["weight"]) or edge["weight"] < 0 or edge["arrival"] <= edge["start"]:
            raise ValueError("workload events must have finite nonnegative costs and positive duration")
        edges.append(edge)
    metadata = dict(payload.get("metadata", {}))
    metadata.update(weight_policy=policy, window=list(window) if window else None, events_filtered=skipped, events_retained=len(edges))
    metadata["cost_semantics"] = {
        "trust-linear": "derived penalty 11-rating; not an observed purchase cost",
        "trust-positive": "derived penalty 11-rating; nonpositive ratings removed",
        "trust-exponential": "derived penalty exp(-rating/5); not an observed purchase cost",
        "recorded-duration": "measured elapsed ride seconds, charged per selected ride; not monetary cost",
        "geographic-distance": "derived haversine station displacement in meters; not actual traveled distance",
    }[policy]
    return {"metadata": metadata, "vertices": list(payload["vertices"]), "edges": edges}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--window", type=float, nargs=2)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    payload = transform_payload(json.loads(raw), args.policy, tuple(args.window) if args.window else None)
    payload["metadata"].update(
        parent_input=args.input.as_posix(),
        parent_sha256=hashlib.sha256(raw).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"{args.output}: {len(payload['edges'])} events; {payload['metadata']['cost_semantics']}")


if __name__ == "__main__":
    main()
