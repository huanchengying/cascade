#!/usr/bin/env python3
"""Oracle checks for shared-star initialized exact masters."""

import unittest

from temporal_mst_exact import exact_mst_w
from temporal_mst_star_priced import star_priced_warm_optimize, star_warm_full
from test_temporal_mst_certificate import fractional_cycle_graph
from test_temporal_mst_cascade_star import cascade_witness
from test_temporal_mst_exact import random_graph, sharing_gap_graph


class StarPricedTests(unittest.TestCase):
    def test_full_and_priced_close_shared_family(self):
        graph = sharing_gap_graph(8)
        full = star_warm_full(graph, "r", time_limit=5)
        priced = star_priced_warm_optimize(graph, "r", time_limit=5)
        self.assertEqual(full["initial_cost"], 108)
        self.assertEqual(full["cost"], 108)
        self.assertEqual(priced["initial_cost"], 108)
        self.assertEqual(priced["cost"], 108)
        self.assertTrue(priced["certified_optimal"])

    def test_random_bounds_and_optima(self):
        for seed in range(25):
            graph = random_graph(140000 + seed, vertex_count=5, edge_count=18)
            optimum = exact_mst_w(graph, "r").cost
            for mode in ("weight", "none", "root", "deadline"):
                with self.subTest(seed=seed, mode=mode):
                    result = star_priced_warm_optimize(
                        graph,
                        "r",
                        time_limit=5,
                        cut_mode=mode,
                        max_probes=20,
                    )
                    self.assertLessEqual(result["lower_bound"], optimum + 1e-6)
                    self.assertAlmostEqual(result["cost"], optimum)
                    self.assertTrue(result["certified_optimal"])

    def test_cascade_incumbent_is_shared_by_full_and_priced_masters(self):
        graph = cascade_witness()
        full = star_warm_full(
            graph,
            "r",
            time_limit=5,
            initial_mode="cascade",
            cascade_rounds=1,
            cascade_max_hubs=20,
            cascade_max_parent_evaluations=100,
        )
        priced = star_priced_warm_optimize(
            graph,
            "r",
            time_limit=5,
            cut_mode="root",
            initial_mode="cascade",
            cascade_rounds=1,
            cascade_max_hubs=20,
            cascade_max_parent_evaluations=100,
        )
        self.assertEqual(full["initial_cost"], 105)
        self.assertEqual(priced["initial_cost"], 105)
        self.assertEqual(full["initial_algorithm"], "batch-relaxed-cascade-star")
        self.assertEqual(priced["initial_algorithm"], "batch-relaxed-cascade-star")
        self.assertEqual(full["cost"], 105)
        self.assertEqual(priced["cost"], 105)

    def test_full_root_precuts_preserve_global_optimum_and_bound(self):
        graph = fractional_cycle_graph()
        result = star_warm_full(
            graph,
            "r",
            time_limit=5,
            cut_mode="root",
            max_probes=20,
        )
        self.assertEqual(result["cost"], 11)
        self.assertAlmostEqual(result["lower_bound"], 11, places=5)
        self.assertGreater(result["root_cuts"], 0)
        self.assertEqual(result["cut_mode"], "root")
        self.assertTrue(result["certified_optimal"])


if __name__ == "__main__":
    unittest.main()
