#!/usr/bin/env python3
"""Determinism and physical-weight checks for weighted dataset converters."""

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from prepare_citibike_weighted import convert_citibike
from temporal_mst_prototype import TemporalGraph


class CitiBikeConversionTests(unittest.TestCase):
    def test_streaming_sample_is_deterministic_and_physical(self) -> None:
        rows = [
            {
                "ride_id": f"ride-{position}",
                "rideable_type": "classic_bike",
                "started_at": f"2026-04-01 00:0{position}:00",
                "ended_at": f"2026-04-01 00:0{position}:30",
                "start_station_name": "A",
                "start_station_id": "A",
                "end_station_name": "B",
                "end_station_id": "B",
                "start_lat": "40.0",
                "start_lng": "-74.0",
                "end_lat": "40.01",
                "end_lng": "-74.01",
                "member_casual": "member",
            }
            for position in range(6)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "trips.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            zip_path = root / "trips.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(csv_path, csv_path.name)
            first = root / "first.json"
            second = root / "second.json"
            convert_citibike(zip_path, first, sample_size=4)
            convert_citibike(zip_path, second, sample_size=4)
            self.assertEqual(first.read_text(), second.read_text())
            payload = json.loads(first.read_text())
            graph = TemporalGraph.from_json(str(first))
            self.assertEqual(len(graph.edges), 4)
            self.assertTrue(all(edge.arrival > edge.start for edge in graph.edges))
            self.assertTrue(all(edge.weight > 1_000 for edge in graph.edges))
            self.assertEqual(
                payload["metadata"]["weight_model"],
                "haversine distance in meters between recorded station coordinates",
            )


if __name__ == "__main__":
    unittest.main()
