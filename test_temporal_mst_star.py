#!/usr/bin/env python3
"""Correctness checks for exact shared-activation star neighborhoods."""

import random
import unittest
from itertools import product

from temporal_mst_exact import exact_mst_w
from temporal_mst_prototype import TemporalEdge, TemporalGraph, temporal_reachable, validate_temporal_tree
from temporal_mst_shared import _parent_exchange_descent, shortest_path_union_tree
from temporal_mst_star import _Fenwick, _TreeIndex, _apply_move, _best_move, shared_activation_star
from test_temporal_mst_exact import random_graph, sharing_gap_graph


class SharedActivationStarTests(unittest.TestCase):
    def test_path_sums_and_ancestor_queries(self):
        tree = [
            TemporalEdge("ra", "r", "a", 0, 1, 1),
            TemporalEdge("ab", "a", "b", 2, 3, 1),
            TemporalEdge("ac", "a", "c", 2, 3, 1),
            TemporalEdge("bd", "b", "d", 4, 5, 1),
        ]
        index = _TreeIndex("r", tree, {"r", "a", "b", "c", "d"})
        fenwick = _Fenwick(5)
        for vertex, value in {"a": 2, "b": 3, "c": 5, "d": 7}.items():
            fenwick.add(index.position[vertex], value)
        self.assertEqual(index.root_path_sum("d", fenwick), 12)
        self.assertEqual(index.root_path_sum("c", fenwick), 7)
        self.assertTrue(index.is_ancestor("a", "d"))
        self.assertFalse(index.is_ancestor("c", "d"))

    def test_fenwick_has_constant_empty_initialization(self):
        fenwick = _Fenwick(10**9)
        self.assertEqual(fenwick.values, {})
        fenwick.add(123456789, 4.5)
        self.assertEqual(fenwick.range_sum(123456789, 123456790), 4.5)
        self.assertLessEqual(len(fenwick.values), 31)

    def test_crosses_joint_activation_barrier(self):
        graph = sharing_gap_graph(8)
        result = shared_activation_star(graph, "r")
        optimum = exact_mst_w(graph, "r").cost
        self.assertEqual(result.baseline_cost, 793)
        self.assertEqual(result.exchange_cost, 793)
        self.assertEqual(result.cost, 108)
        self.assertEqual(result.cost, optimum)
        self.assertEqual(result.accepted_moves, 1)
        self.assertEqual(result.activated_children, 8)

    def test_random_results_are_feasible_and_monotone(self):
        rng = random.Random(20261008)
        for iteration in range(80):
            graph = random_graph(
                rng.randrange(10_000_000), vertex_count=7, edge_count=24
            )
            with self.subTest(iteration=iteration):
                result = shared_activation_star(graph, "r", max_rounds=4)
                selected = [edge for edge in graph.edges if edge.eid in result.edge_ids]
                required = temporal_reachable(graph, "r", graph.edges)
                self.assertLessEqual(result.cost, result.exchange_cost + 1e-7)
                self.assertLessEqual(result.exchange_cost, result.baseline_cost + 1e-7)
                self.assertTrue(validate_temporal_tree(graph, "r", selected, required))

    def test_best_move_matches_exhaustive_star_neighborhood(self):
        rng = random.Random(2026100801)
        for iteration in range(35):
            graph = random_graph(
                rng.randrange(10_000_000), vertex_count=5, edge_count=16
            )
            required = temporal_reachable(graph, "r", graph.edges)
            _, baseline, _ = shortest_path_union_tree(graph, "r")
            tree, _ = _parent_exchange_descent(graph, "r", required, baseline)
            index = _TreeIndex("r", tree, required)
            current_cost = sum(edge.weight for edge in tree)
            exhaustive = current_cost
            for hub in required - {"r"}:
                parents = [
                    edge for edge in graph.edges
                    if edge.v == hub
                    and edge.u in required
                    and index.arrival[edge.u] <= edge.start
                    and not index.is_ancestor(hub, edge.u)
                    and edge.arrival <= index.deadline[hub]
                ]
                for parent_edge in parents:
                    choices = []
                    targets = sorted(required - {"r", hub})
                    for target in targets:
                        options = [index.parent_edge[target]]
                        options.extend(
                            edge for edge in graph.edges
                            if edge.u == hub
                            and edge.v == target
                            and edge.start >= parent_edge.arrival
                            and edge.arrival <= index.deadline[target]
                            and not index.is_ancestor(target, parent_edge.u)
                        )
                        choices.append(options)
                    for selected in product(*choices):
                        parent = dict(index.parent_edge)
                        parent[hub] = parent_edge
                        parent.update(zip(targets, selected))
                        candidate = list(parent.values())
                        if validate_temporal_tree(graph, "r", candidate, required):
                            exhaustive = min(
                                exhaustive, sum(edge.weight for edge in candidate)
                            )
            move, _, _ = _best_move(graph, "r", required, tree)
            if move is None:
                swept = current_cost
            else:
                swept, _, _ = _apply_move(
                    graph, "r", required, tree, move
                )
            with self.subTest(iteration=iteration):
                self.assertAlmostEqual(swept, exhaustive)

    def test_rejects_zero_duration(self):
        graph = TemporalGraph(
            ["r", "a"], [TemporalEdge("ra", "r", "a", 0, 0, 1)]
        )
        with self.assertRaisesRegex(ValueError, "positive durations"):
            shared_activation_star(graph, "r")


if __name__ == "__main__":
    unittest.main()
