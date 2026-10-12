#!/usr/bin/env python3
"""Determinism and provenance checks for the NYC TLC converter."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet

from prepare_tlc_taxi_weighted import convert_tlc_taxi
from temporal_mst_prototype import TemporalGraph


class TLCTaxiConversionTests(unittest.TestCase):
    def test_sample_and_weight_variants_are_deterministic(self) -> None:
        start = datetime(2025, 1, 2, 8)
        table = arrow.table(
            {
                "tpep_pickup_datetime": [start + timedelta(minutes=i) for i in range(6)],
                "tpep_dropoff_datetime": [
                    start + timedelta(minutes=i + 10) for i in range(6)
                ],
                "PULocationID": [1, 2, 3, 4, 5, 6],
                "DOLocationID": [2, 3, 4, 5, 5, 7],
                "trip_distance": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                "total_amount": [12.0, 13.0, 14.0, -1.0, 16.0, 17.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "tiny.parquet"
            parquet.write_table(table, raw, row_group_size=2)
            first = convert_tlc_taxi(raw, root / "first", sample_size=3, batch_size=2)
            second = convert_tlc_taxi(raw, root / "second", sample_size=3, batch_size=3)

            for name in first:
                self.assertEqual(first[name].read_text(), second[name].read_text())
            payloads = {name: json.loads(path.read_text()) for name, path in first.items()}
            edge_ids = {
                name: [edge["id"] for edge in payload["edges"]]
                for name, payload in payloads.items()
            }
            self.assertEqual(edge_ids["duration"], edge_ids["distance"])
            self.assertEqual(edge_ids["duration"], edge_ids["total-amount"])
            self.assertEqual(
                [edge["weight"] for edge in payloads["duration"]["edges"]],
                [edge["duration_s"] for edge in payloads["duration"]["edges"]],
            )
            self.assertEqual(
                payloads["duration"]["metadata"]["rejection_counts"],
                {"invalid_total_amount": 1, "same_taxi_zone": 1},
            )
            graph = TemporalGraph.from_json(str(first["total-amount"]))
            self.assertEqual(len(graph.edges), 3)
            self.assertTrue(all(edge.arrival > edge.start for edge in graph.edges))
            self.assertTrue(all(edge.weight > 0 for edge in graph.edges))

    def test_rejects_empty_sample_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_size"):
            convert_tlc_taxi(Path("missing.parquet"), Path("out"), sample_size=0)


if __name__ == "__main__":
    unittest.main()
