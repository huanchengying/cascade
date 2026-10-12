#!/usr/bin/env python3
"""Independent-oracle checks for warm priced masters and bound scope."""

import unittest

from temporal_mst_exact import exact_mst_w
from temporal_mst_priced import priced_warm_optimize
from test_temporal_mst_exact import random_graph, sharing_gap_graph


class PricedTests(unittest.TestCase):
    def test_shared_cost_family(self):
        result = priced_warm_optimize(sharing_gap_graph(8), "r", time_limit=5)
        self.assertEqual(result["cost"], 108)
        self.assertTrue(result["certified_optimal"])

    def test_random_bounds_and_optima(self):
        for seed in range(30):
            graph = random_graph(120000 + seed, vertex_count=5, edge_count=18)
            optimum = exact_mst_w(graph, "r").cost
            for mode in ("weight", "none", "root", "deadline"):
                with self.subTest(seed=seed, mode=mode):
                    result = priced_warm_optimize(graph, "r", time_limit=5, cut_mode=mode, max_probes=20)
                    self.assertLessEqual(result["lower_bound"], optimum + 1e-6)
                    self.assertAlmostEqual(result["cost"], optimum)
                    self.assertTrue(result["certified_optimal"])

    def test_restricted_bounds_not_promoted(self):
        graph = random_graph(81445, vertex_count=6, edge_count=24)
        optimum = exact_mst_w(graph, "r").cost
        result = priced_warm_optimize(graph, "r", time_limit=5, safe_master_limit=0)
        self.assertLessEqual(result["lower_bound"], optimum + 1e-6)


if __name__ == "__main__":
    unittest.main()
