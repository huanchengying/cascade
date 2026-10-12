import unittest
from unittest.mock import Mock, patch

import highspy

from temporal_mst_exact import exact_mst_w
from temporal_mst_highs import ikuta_warm_start
from test_temporal_mst_exact import random_graph, sharing_gap_graph


class NativeHighsTests(unittest.TestCase):
    def test_warm_start_matches_oracle(self):
        for seed in range(30):
            graph = random_graph(seed + 92200, vertex_count=5, edge_count=14)
            result = ikuta_warm_start(graph, "r", time_limit=5)
            self.assertTrue(result["optimal"])
            self.assertAlmostEqual(result["cost"], exact_mst_w(graph, "r").cost)

    def test_shared_cost_and_validated_initial_tree(self):
        graph = sharing_gap_graph(8)
        result = ikuta_warm_start(graph, "r", time_limit=5)
        self.assertEqual(result["cost"], 108)
        with self.assertRaisesRegex(ValueError, "valid spanning"):
            ikuta_warm_start(graph, "r", initial_ids=["late_hub"])

    def test_solution_vector_copied_once(self):
        reads = []
        native = highspy.Highs()
        wrapper = Mock(wraps=native)

        class CountedSolution:
            def __init__(self, solution):
                self.solution = solution
                self.value_valid = solution.value_valid

            @property
            def col_value(self):
                reads.append(True)
                return self.solution.col_value

        wrapper.getSolution.side_effect = lambda: CountedSolution(native.getSolution())
        with patch("temporal_mst_highs.highspy.Highs", return_value=wrapper):
            result = ikuta_warm_start(sharing_gap_graph(20), "r", time_limit=5)
        self.assertEqual(result["cost"], 120)
        self.assertEqual(len(reads), 1)


if __name__ == "__main__":
    unittest.main()
