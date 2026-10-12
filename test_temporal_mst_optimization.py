#!/usr/bin/env python3
"""Regression tests for the Ikuta ILP and shared-cost optimizer."""

import importlib.util
import random
import unittest

from temporal_mst_exact import exact_mst_w
from temporal_mst_ilp import ikuta_ilp
from temporal_mst_lagrangian import shared_availability_lagrangian
from temporal_mst_lns import exact_region_reconnection, shared_cost_lns
from temporal_mst_prototype import (
    TemporalEdge,
    TemporalGraph,
    exact_temporal_tree,
    temporal_reachable,
    validate_temporal_tree,
)
from temporal_mst_shared import shared_prefix_optimize
from test_temporal_mst_exact import random_graph, sharing_gap_graph


SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


def cut_required_graph() -> TemporalGraph:
    return TemporalGraph(
        ["r", "a", "b"],
        [
            TemporalEdge("root_a", "r", "a", 0, 0, 10),
            TemporalEdge("root_b", "r", "b", 0, 0, 10),
            TemporalEdge("a_b", "a", "b", 0, 0, 1),
            TemporalEdge("b_a", "b", "a", 0, 0, 1),
        ],
    )


class IkutaILPTests(unittest.TestCase):
    @unittest.skipUnless(SCIPY_AVAILABLE, "SciPy MILP backend is optional")
    def test_sharing_gap_matches_subset_dp(self) -> None:
        graph = sharing_gap_graph(4)
        result = ikuta_ilp(graph, "r")
        exact = exact_mst_w(graph, "r", backend="implicit")
        self.assertTrue(result.optimal)
        self.assertTrue(result.cuts_proven_redundant)
        self.assertEqual(result.connectivity_mode, "cut-free-positive-duration")
        self.assertEqual(result.cuts_added, 0)
        self.assertEqual(result.cost, exact.cost)
        selected = [edge for edge in graph.edges if edge.eid in result.edge_ids]
        self.assertTrue(validate_temporal_tree(graph, "r", selected, set(result.reachable)))

    @unittest.skipUnless(SCIPY_AVAILABLE, "SciPy MILP backend is optional")
    def test_connectivity_cut_excludes_zero_time_cycle(self) -> None:
        graph = cut_required_graph()
        result = ikuta_ilp(graph, "r")
        self.assertEqual(result.cost, 11)
        self.assertFalse(result.cuts_proven_redundant)
        self.assertEqual(result.connectivity_mode, "iterative-connectivity-cuts")
        self.assertGreaterEqual(result.cuts_added, 1)
        self.assertGreaterEqual(result.rounds, 2)

    @unittest.skipUnless(SCIPY_AVAILABLE, "SciPy MILP backend is optional")
    def test_random_instances_match_edge_subset_oracle(self) -> None:
        for seed in range(20):
            graph = random_graph(seed + 10_000, vertex_count=4, edge_count=9)
            expected, _ = exact_temporal_tree(graph, "r")
            with self.subTest(seed=seed):
                result = ikuta_ilp(graph, "r")
                self.assertAlmostEqual(result.cost, expected)


class SharedPrefixTests(unittest.TestCase):
    def test_batch_move_closes_path_union_sharing_gap(self) -> None:
        graph = sharing_gap_graph(8)
        result = shared_prefix_optimize(graph, "r")
        optimum = exact_mst_w(graph, "r", backend="implicit").cost
        self.assertEqual(result.baseline_cost, 793)
        self.assertEqual(result.cost, 108)
        self.assertEqual(result.cost, optimum)
        self.assertGreater(result.relative_improvement, 0.8)
        self.assertGreaterEqual(result.accepted_moves, 1)

    def test_random_instances_remain_feasible_and_monotone(self) -> None:
        for seed in range(30):
            graph = random_graph(seed + 20_000, vertex_count=6, edge_count=16)
            if any(edge.arrival <= edge.start for edge in graph.edges):
                continue
            with self.subTest(seed=seed):
                result = shared_prefix_optimize(
                    graph,
                    "r",
                    max_rounds=2,
                    max_hub_states=12,
                    max_states_per_vertex=3,
                    evaluated_prefixes=4,
                )
                self.assertLessEqual(result.cost, result.baseline_cost + 1e-9)
                selected = [edge for edge in graph.edges if edge.eid in result.edge_ids]
                required = temporal_reachable(graph, "r", graph.edges)
                self.assertTrue(validate_temporal_tree(graph, "r", selected, required))

    def test_requires_strictly_positive_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive durations"):
            shared_prefix_optimize(cut_required_graph(), "r")


class SharedLNSTests(unittest.TestCase):
    def test_exact_region_reconnection_matches_global_optimum(self) -> None:
        graph = sharing_gap_graph(5)
        baseline = shared_prefix_optimize(graph, "r", max_rounds=0)
        tree = [edge for edge in graph.edges if edge.eid in baseline.edge_ids]
        cost, candidate, _ = exact_region_reconnection(
            graph,
            "r",
            tree,
            set(graph.vertices) - {"r"},
        )
        optimum = exact_mst_w(graph, "r", backend="implicit").cost
        self.assertEqual(cost, optimum)
        self.assertTrue(validate_temporal_tree(graph, "r", candidate, set(graph.vertices)))

    def test_lns_crosses_joint_activation_barrier(self) -> None:
        graph = sharing_gap_graph(8)
        result = shared_cost_lns(
            graph,
            "r",
            max_neighborhood=9,
            max_hub_states=16,
        )
        self.assertEqual(result.baseline_cost, 793)
        self.assertEqual(result.cost, 108)
        self.assertGreaterEqual(result.accepted_neighborhoods, 1)

    def test_lns_is_feasible_and_monotone(self) -> None:
        rng = random.Random(20260914)
        for iteration in range(15):
            graph = random_graph(
                rng.randrange(1_000_000), vertex_count=6, edge_count=16
            )
            with self.subTest(iteration=iteration):
                result = shared_cost_lns(
                    graph,
                    "r",
                    max_rounds=1,
                    max_hub_states=8,
                    max_neighborhood=5,
                    max_dp_cells=500_000,
                )
                selected = [edge for edge in graph.edges if edge.eid in result.edge_ids]
                required = temporal_reachable(graph, "r", graph.edges)
                self.assertLessEqual(result.cost, result.baseline_cost + 1e-9)
                self.assertTrue(validate_temporal_tree(graph, "r", selected, required))


class LagrangianTests(unittest.TestCase):
    def test_shared_credit_closes_gap_family(self) -> None:
        graph = sharing_gap_graph(8)
        result = shared_availability_lagrangian(
            graph,
            "r",
            iterations=500,
            primal_interval=5,
        )
        self.assertEqual(result.cost, 108)
        self.assertAlmostEqual(result.lower_bound, 108, places=5)
        self.assertAlmostEqual(result.relative_gap, 0, places=7)
        self.assertTrue(result.certified_optimal)
        self.assertLessEqual(result.safe_edge_count, result.original_edge_count)

    def test_random_bounds_are_valid(self) -> None:
        for seed in range(20):
            graph = random_graph(seed + 30_000, vertex_count=5, edge_count=12)
            optimum = exact_mst_w(graph, "r", backend="implicit").cost
            with self.subTest(seed=seed):
                result = shared_availability_lagrangian(
                    graph,
                    "r",
                    iterations=100,
                    primal_interval=10,
                )
                self.assertLessEqual(result.lower_bound, optimum + 1e-7)
                self.assertGreaterEqual(result.cost + 1e-7, optimum)
                if result.certified_optimal:
                    self.assertAlmostEqual(result.cost, optimum)
                selected = [edge for edge in graph.edges if edge.eid in result.edge_ids]
                required = temporal_reachable(graph, "r", graph.edges)
                self.assertTrue(validate_temporal_tree(graph, "r", selected, required))

    def test_reduced_cost_fixing_preserves_optimum(self) -> None:
        graph = random_graph(40_000, vertex_count=6, edge_count=18)
        optimum = exact_mst_w(graph, "r", backend="implicit").cost
        result = shared_availability_lagrangian(
            graph,
            "r",
            iterations=100,
            primal_interval=10,
            pool_width=0,
            pool_time_limit=0,
            safe_master_time_limit=10,
        )
        self.assertGreater(result.safe_reduction_ratio, 0)
        self.assertTrue(result.certified_optimal)
        self.assertAlmostEqual(result.cost, optimum)


if __name__ == "__main__":
    unittest.main()
