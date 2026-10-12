import unittest

from prepare_root_selection_20261008 import select_roots
from temporal_mst_prototype import TemporalEdge, TemporalGraph


class RootSelectionTests(unittest.TestCase):
    def test_reachability_then_outdegree_then_identifier(self):
        graph = TemporalGraph(
            ["a", "b", "c", "d"],
            [
                TemporalEdge("a-b-1", "a", "b", 1, 2, 1),
                TemporalEdge("a-b-2", "a", "b", 2, 3, 1),
                TemporalEdge("a-b-3", "a", "b", 3, 4, 1),
                TemporalEdge("b-d", "b", "d", 0, 1, 1),
                TemporalEdge("c-a", "c", "a", 0, 1, 1),
            ],
        )
        selected, ranking = select_roots(graph, candidate_count=3, selection_count=2)
        self.assertEqual([row["root"] for row in selected], ["c", "a"])
        self.assertEqual([row["root"] for row in ranking], ["c", "a", "b"])

    def test_invalid_counts(self):
        graph = TemporalGraph([], [])
        with self.assertRaises(ValueError):
            select_roots(graph, candidate_count=2, selection_count=3)


if __name__ == "__main__":
    unittest.main()
