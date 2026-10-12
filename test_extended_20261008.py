import math
import unittest

from benchmark_extended_20261008 import earliest_weight_branching, measured_kernel, snapshot_mst
from temporal_mst_cascade_star import cascade_activation_star
from temporal_mst_prototype import TemporalEdge, TemporalGraph, temporal_reachable, validate_temporal_tree
from test_temporal_mst_cascade_star import cascade_witness


class ExtendedKernelTests(unittest.TestCase):
    def test_earliest_branching_uses_minimum_weight_compatible_ties(self):
        graph = TemporalGraph(['r', 'a', 'b'], [TemporalEdge('early-expensive', 'r', 'a', 0, 1, 5), TemporalEdge('early-cheap', 'r', 'a', 0, 1, 2), TemporalEdge('late', 'r', 'a', 0, 3, 1), TemporalEdge('early-child', 'a', 'b', 1, 2, 2), TemporalEdge('late-child', 'a', 'b', 3, 4, 1)])
        selected = earliest_weight_branching(graph, 'r')
        self.assertEqual({edge.eid for edge in selected}, {'early-cheap', 'early-child'})
        self.assertTrue(validate_temporal_tree(graph, 'r', selected, {'r', 'a', 'b'}))

    def test_snapshot_tree_can_break_chronology_despite_connected_topology(self):
        graph = TemporalGraph(['r', 'a', 'b'], [TemporalEdge('slow', 'r', 'a', 0, 5, 1), TemporalEdge('missed', 'a', 'b', 2, 3, 1), TemporalEdge('direct', 'r', 'b', 0, 4, 4)])
        selected = snapshot_mst(graph, {'r', 'a', 'b'})
        self.assertEqual(len(selected), 2)
        self.assertFalse(validate_temporal_tree(graph, 'r', selected, {'r', 'a', 'b'}))
        self.assertEqual(temporal_reachable(graph, 'r', selected), {'r', 'a'})

    def test_phase_instrumentation_preserves_original_cascade_result(self):
        graph = cascade_witness()
        expected = cascade_activation_star(graph, 'r')
        measured, elapsed, phases, calls = measured_kernel(graph, 'r', 'cascade', {})
        self.assertEqual(measured['cost'], expected.cost)
        self.assertEqual(set(measured['edge_ids']), set(expected.edge_ids))
        self.assertLessEqual(math.fsum(phases.values()), elapsed + 1e-6)
        self.assertGreater(calls['fixed_parent_dp'], 0)


if __name__ == '__main__':
    unittest.main()
