#!/usr/bin/env python3
"""Projection and oracle checks for full time-state prefix cuts."""

import unittest

import numpy as np

from temporal_mst_certificate import ConnectivityCut, build_prefix_model, cut_row, solve_prefix_lp
from temporal_mst_exact import exact_mst_w
from temporal_mst_prototype import TemporalEdge, TemporalGraph
from temporal_mst_state_cuts import (
    StateExpansion,
    StatePrefixCut,
    solve_complete_uniform_deadline_lp,
    solve_state_prefix_lp,
    state_cut_row,
)
from test_temporal_mst_exact import random_graph


def deadline_witness_graph():
    return TemporalGraph(["r", "a", "b", "c"], [
        TemporalEdge("ra", "r", "a", 3, 4, 9),
        TemporalEdge("ab", "a", "b", 5, 6, 7),
        TemporalEdge("ba", "b", "a", 8, 9, 6),
        TemporalEdge("rb", "r", "b", 7, 10, 6),
        TemporalEdge("ac", "a", "c", 9, 13, 7),
    ])


class StatePrefixCutTests(unittest.TestCase):
    def test_uniform_deadline_cut_is_a_state_prefix_cut(self):
        model = build_prefix_model(deadline_witness_graph(), "r")
        deadline = ConnectivityCut(frozenset({"a", "b"}), "a", 9)
        uniform = StatePrefixCut((("a", 9), ("b", 6)), "a", 9)
        compact_row, compact_bound = cut_row(model, deadline)
        state_row, state_bound = state_cut_row(StateExpansion(model), uniform)
        self.assertEqual(compact_bound, state_bound)
        self.assertEqual(compact_row, state_row)

    def test_state_cuts_close_deadline_witness(self):
        graph = deadline_witness_graph()
        model = build_prefix_model(graph, "r")
        root = solve_prefix_lp(model, cut_mode="root", max_rounds=8, max_probes=20)
        deadline = solve_prefix_lp(model, cut_mode="deadline", max_rounds=8, max_probes=20)
        state = solve_state_prefix_lp(model, max_rounds=8, max_probes=20)
        optimum = exact_mst_w(graph, "r").cost
        self.assertLess(root.lp_objective, deadline.lp_objective)
        self.assertAlmostEqual(deadline.lp_objective, optimum)
        self.assertAlmostEqual(state.lp_objective, optimum)

    def test_exhaustive_uniform_separation_closes_missed_probe(self):
        graph = random_graph(200039, vertex_count=5, edge_count=22)
        model = build_prefix_model(graph, "r")
        capped = solve_prefix_lp(
            model, cut_mode="deadline", max_rounds=8, max_probes=30
        )
        complete = solve_complete_uniform_deadline_lp(model)
        state = solve_state_prefix_lp(
            model, max_rounds=20, max_probes=30, all_prefixes=True
        )
        self.assertAlmostEqual(capped.lp_objective, 14)
        self.assertAlmostEqual(complete.lp_objective, 16)
        self.assertAlmostEqual(state.lp_objective, 16)

    def test_state_prefix_strictly_dominates_uniform_family(self):
        graph = TemporalGraph(["r", "a", "b"], [
            TemporalEdge("ra", "r", "a", 1, 2, 8),
            TemporalEdge("ab", "a", "b", 6, 7, 4),
            TemporalEdge("rb", "r", "b", 8, 9, 6),
            TemporalEdge("ba", "b", "a", 7, 10, 0),
        ])
        model = build_prefix_model(graph, "r")
        uniform = solve_complete_uniform_deadline_lp(model, max_rounds=30)
        state = solve_state_prefix_lp(
            model, max_rounds=30, max_probes=100, all_prefixes=True
        )
        self.assertAlmostEqual(uniform.lp_objective, 9)
        self.assertAlmostEqual(state.lp_objective, 12)
        self.assertAlmostEqual(exact_mst_w(graph, "r").cost, 12)

    def test_random_state_bounds_are_valid(self):
        for seed in range(35):
            graph = random_graph(150000 + seed, vertex_count=5, edge_count=18)
            model = build_prefix_model(graph, "r")
            state = solve_state_prefix_lp(model, max_rounds=6, max_probes=20)
            optimum = exact_mst_w(graph, "r").cost
            with self.subTest(seed=seed):
                self.assertLessEqual(state.lower_bound, optimum + 1e-6)
                self.assertTrue(np.all(state.fractions >= -1e-8))


if __name__ == "__main__":
    unittest.main()
