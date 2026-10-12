#!/usr/bin/env python3
"""Regression tests for weighted benchmark result persistence."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_weighted_mst
from temporal_mst_prototype import TemporalEdge, TemporalGraph


class WeightedBenchmarkTests(unittest.TestCase):
    def test_ikuta_timeout_is_recorded_and_checkpointed(self) -> None:
        graph = TemporalGraph(
            ["r", "a"],
            [TemporalEdge("edge", "r", "a", 0, 1, 1)],
        )

        def fake_run_method(work_graph, root, method, time_limit):
            if method == "ikuta_ilp":
                raise TimeoutError("test timeout")
            return {
                "method": method,
                "cost": 1.0,
                "edge_ids": ["edge"],
                "wall_time_s": 0.01,
                "vertices": len(work_graph.vertices),
                "edges": len(work_graph.edges),
                "reachable": 2,
                "incoming_lower_bound": 1.0,
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            argv = [
                "benchmark_weighted_mst.py",
                "input.json",
                "--root",
                "r",
                "--full-ilp",
                "--output",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    benchmark_weighted_mst.TemporalGraph,
                    "from_json",
                    return_value=graph,
                ),
                patch.object(
                    benchmark_weighted_mst,
                    "induced_temporal_sample",
                    return_value=graph,
                ),
                patch.object(
                    benchmark_weighted_mst,
                    "run_method",
                    side_effect=fake_run_method,
                ),
                patch.object(
                    benchmark_weighted_mst,
                    "write_results",
                    wraps=benchmark_weighted_mst.write_results,
                ) as write_results,
            ):
                benchmark_weighted_mst.main()

            payload = json.loads(output.read_text())
            timeout_records = [
                record
                for record in payload["records"]
                if record.get("status") == "timeout-no-incumbent"
            ]
            self.assertEqual(len(payload["records"]), 10)
            self.assertEqual(len(timeout_records), 2)
            self.assertEqual({record["scope"] for record in timeout_records}, {"exact_sample", "full"})
            self.assertTrue(all(record["method"] == "ikuta_ilp" for record in timeout_records))
            self.assertEqual(write_results.call_count, 10)
            self.assertTrue(output.with_suffix(".csv").exists())


if __name__ == "__main__":
    unittest.main()
