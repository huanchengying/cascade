import unittest
from pathlib import Path

from temporal_mst_exact import exact_mst_w
from temporal_mst_prototype import TemporalEdge, TemporalGraph
from temporal_mst_speedsteiner import speedsteiner_tree
from test_temporal_mst_exact import random_graph, sharing_gap_graph


@unittest.skipUnless(Path("work/bin/speedsteiner_adapter").is_file(), "optional author binary not built")
class SpeedSteinerTests(unittest.TestCase):
    def test_sharing_family(self):
        result = speedsteiner_tree(sharing_gap_graph(8), "r", time_limit=5)
        self.assertEqual(result["cost"], 108)

    def test_parallel_state_arcs_choose_minimum(self):
        graph = TemporalGraph(["r", "a"], [TemporalEdge("bad", "r", "a", 0, 1, 10), TemporalEdge("good", "r", "a", 0, 1, 1)])
        self.assertEqual(speedsteiner_tree(graph, "r")["edge_ids"], ["good"])

    def test_fast_and_level_two_agree_on_cost(self):
        for seed in range(30):
            graph = random_graph(130000 + seed, vertex_count=5, edge_count=18)
            fast = speedsteiner_tree(graph, "r", time_limit=5)
            plain = speedsteiner_tree(graph, "r", time_limit=5, method="level2")
            with self.subTest(seed=seed):
                self.assertAlmostEqual(fast["cost"], plain["cost"])
                self.assertGreaterEqual(fast["cost"] + 1e-7, exact_mst_w(graph, "r").cost)


if __name__ == "__main__":
    unittest.main()
