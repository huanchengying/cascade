#!/usr/bin/env python3
"""LP cut validity, safe pricing, pool closure and oracle differential tests."""

import itertools
import math
import unittest

import numpy as np

from temporal_mst_certificate import (
    ConnectivityCut, build_prefix_model, certified_pool_optimize, cut_row,
    safe_event_ids, solve_prefix_lp, support_closed_pool,
)
from temporal_mst_exact import exact_mst_w
from temporal_mst_ilp import ikuta_ilp
from temporal_mst_prototype import TemporalEdge, TemporalGraph, validate_temporal_tree
from test_temporal_mst_exact import random_graph, sharing_gap_graph


def fractional_cycle_graph():
    return TemporalGraph(["r", "a", "b"], [
        TemporalEdge("root", "r", "a", 0, 1, 10),
        TemporalEdge("early", "a", "b", 1, 2, 1),
        TemporalEdge("return", "b", "a", 3, 4, 1),
        TemporalEdge("late", "a", "b", 4, 5, 1),
    ])


class CertificateTests(unittest.TestCase):
    def test_positive_duration_cuts_are_not_lp_redundant(self):
        model = build_prefix_model(fractional_cycle_graph(), "r")
        plain = solve_prefix_lp(model, cut_mode="none")
        rooted = solve_prefix_lp(model, cut_mode="root", max_probes=20)
        self.assertAlmostEqual(plain.lp_objective, 6.5)
        self.assertAlmostEqual(rooted.lp_objective, 11)
        self.assertGreater(len(rooted.cuts), 0)
        self.assertEqual(ikuta_ilp(model.graph, "r").cost, 11)

    def test_all_deadline_cuts_valid_on_every_feasible_tree(self):
        graph = fractional_cycle_graph()
        model = build_prefix_model(graph, "r")
        for chosen in itertools.combinations(model.edges, 2):
            if not validate_temporal_tree(graph, "r", chosen, set(graph.vertices)):
                continue
            chosen_ids = {edge.eid for edge in chosen}
            values = np.zeros(2 * len(model.edges))
            values[:len(model.edges)] = [float(edge.eid in chosen_ids) for edge in model.edges]
            for positions in model.incoming.values():
                mass = 0
                for position in positions:
                    mass += values[position]
                    values[len(model.edges) + position] = mass
            for side in (frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})):
                for target in side:
                    for deadline in (1, 2, 3, 4, 5):
                        coefficients, rhs = cut_row(model, ConnectivityCut(side, target, deadline))
                        self.assertLessEqual(sum(value * values[position] for position, value in coefficients.items()), rhs + 1e-10)

    def test_shared_prefix_family_is_certified(self):
        graph = sharing_gap_graph(8)
        result = certified_pool_optimize(graph, "r", time_limit=10)
        self.assertEqual(result.cost, 108)
        self.assertTrue(result.certified_optimal)
        self.assertAlmostEqual(result.lower_bound, 108, places=4)

    def test_deadline_cut_is_stronger_than_root_cuts(self):
        graph = random_graph(90082, vertex_count=5, edge_count=22)
        model = build_prefix_model(graph, "r")
        rooted = solve_prefix_lp(model, cut_mode="root", max_rounds=8, max_probes=20)
        deadline = solve_prefix_lp(model, cut_mode="deadline", max_rounds=8, max_probes=20)
        self.assertAlmostEqual(rooted.lp_objective, 21)
        self.assertAlmostEqual(deadline.lp_objective, 22)
        self.assertEqual(exact_mst_w(graph, "r").cost, 22)

    def test_minimal_deadline_witness(self):
        graph = TemporalGraph(["r", "a", "b", "c"], [
            TemporalEdge("ra", "r", "a", 3, 4, 9), TemporalEdge("ab", "a", "b", 5, 6, 7),
            TemporalEdge("ba", "b", "a", 8, 9, 6), TemporalEdge("rb", "r", "b", 7, 10, 6),
            TemporalEdge("ac", "a", "c", 9, 13, 7),
        ])
        model = build_prefix_model(graph, "r")
        self.assertAlmostEqual(solve_prefix_lp(model, cut_mode="root").lp_objective, 21)
        self.assertAlmostEqual(solve_prefix_lp(model, cut_mode="deadline").lp_objective, 22)
        self.assertEqual(exact_mst_w(graph, "r").cost, 22)

    def test_ikuta_accepts_fractional_separation_cuts(self):
        graph = fractional_cycle_graph()
        model = build_prefix_model(graph, "r")
        lp = solve_prefix_lp(model, cut_mode="root")
        result = ikuta_ilp(graph, "r", initial_cut_sets=[cut.side for cut in lp.cuts])
        self.assertEqual(result.cost, 11)
        self.assertGreater(result.cuts_added, 0)

    def test_random_safe_sets_preserve_optimum(self):
        for seed in range(80):
            graph = random_graph(seed + 80000, vertex_count=5, edge_count=14)
            optimum = exact_mst_w(graph, "r").cost
            model = build_prefix_model(graph, "r")
            lp = solve_prefix_lp(model, cut_mode="deadline", max_rounds=3, max_probes=20)
            incumbent = ikuta_ilp(graph, "r")
            safe_ids = safe_event_ids(model, lp, incumbent.cost, incumbent.edge_ids)
            safe_graph = TemporalGraph(graph.vertices, [edge for edge in model.edges if edge.eid in safe_ids])
            with self.subTest(seed=seed):
                self.assertLessEqual(lp.lower_bound, optimum + 1e-7)
                self.assertAlmostEqual(ikuta_ilp(safe_graph, "r").cost, optimum)
                if model.edges:
                    pool = support_closed_pool(model, lp, safe_ids, incumbent.edge_ids, 1)
                    self.assertTrue(set(incumbent.edge_ids) <= pool <= safe_ids)
                    for edge in model.edges:
                        if edge.eid in pool and edge.u != "r":
                            self.assertTrue(any(support.eid in pool and support.v == edge.u and support.arrival <= edge.start for support in model.edges))

    def test_restricted_bound_is_not_global(self):
        graph = random_graph(81445, vertex_count=6, edge_count=24)
        optimum = exact_mst_w(graph, "r").cost
        result = certified_pool_optimize(graph, "r", time_limit=10, safe_master_limit=0)
        self.assertLessEqual(result.lower_bound, optimum + 1e-7)
        self.assertGreaterEqual(result.cost + 1e-7, optimum)
        if result.certified_optimal:
            self.assertAlmostEqual(result.cost, optimum)

    def test_invalid_settings_and_zero_duration(self):
        with self.assertRaisesRegex(ValueError, "positive durations"):
            certified_pool_optimize(TemporalGraph(["r", "a"], [TemporalEdge("zero", "r", "a", 1, 1, 1)]), "r")
        with self.assertRaises(ValueError):
            certified_pool_optimize(fractional_cycle_graph(), "r", time_limit=0)
        result = certified_pool_optimize(TemporalGraph(["r"], []), "r")
        self.assertEqual(result.cost, 0)
        self.assertTrue(result.certified_optimal)


if __name__ == "__main__":
    unittest.main()
