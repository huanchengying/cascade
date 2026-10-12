#!/usr/bin/env python3
"""Reference-density search and temporal reconstruction checks."""

import unittest

from temporal_mst_exact import exact_mst_w
from temporal_mst_huang import huang_recursive_greedy
from temporal_mst_prototype import TemporalGraph, validate_temporal_tree
from temporal_mst_shared import shortest_path_union_tree
from test_temporal_mst_exact import random_graph, sharing_gap_graph


class HuangTests(unittest.TestCase):
    def test_level_one_matches_independent_path_union(self):
        for seed in range(30):
            graph = random_graph(seed + 90100, vertex_count=5, edge_count=12)
            result = huang_recursive_greedy(graph, "r", level=1)
            expected, _, _ = shortest_path_union_tree(graph, "r")
            self.assertAlmostEqual(result.cost, expected)

    def test_recursive_sharing_family(self):
        graph = sharing_gap_graph(4)
        self.assertEqual(huang_recursive_greedy(graph, "r", level=1).cost, 397)
        for level in (2, 3):
            result = huang_recursive_greedy(graph, "r", level=level)
            self.assertEqual(result.cost, 104)

    def test_random_trees_are_feasible(self):
        for seed in range(30):
            graph = random_graph(seed + 90200, vertex_count=5, edge_count=14)
            optimum = exact_mst_w(graph, "r").cost
            for level in (2, 3):
                result = huang_recursive_greedy(graph, "r", level=level)
                chosen = [edge for edge in graph.edges if edge.eid in result.edge_ids]
                self.assertTrue(validate_temporal_tree(graph, "r", chosen, set(result.reachable)))
                self.assertGreaterEqual(result.cost + 1e-9, optimum)

    def test_resource_guard_and_empty(self):
        with self.assertRaisesRegex(ValueError, "metric closure guard"):
            huang_recursive_greedy(sharing_gap_graph(4), "r", max_states=1)
        self.assertEqual(huang_recursive_greedy(TemporalGraph(["r"], []), "r").cost, 0)

    def test_improved_search_matches_naive_algorithm_three(self):
        for seed in range(40):
            graph = random_graph(seed + 90700, vertex_count=5, edge_count=14)
            for level in (2, 3):
                improved = huang_recursive_greedy(graph, "r", level=level)
                naive = huang_recursive_greedy(graph, "r", level=level, search="naive")
                with self.subTest(seed=seed, level=level):
                    self.assertAlmostEqual(improved.closure_cost, naive.closure_cost)
                    self.assertAlmostEqual(improved.cost, naive.cost)


if __name__ == "__main__":
    unittest.main()
