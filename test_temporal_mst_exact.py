#!/usr/bin/env python3
"""Differential correctness, counterexamples, and reconstruction regressions."""

import math
import random
import unittest
from unittest.mock import patch

import numpy as np

from temporal_dag_framework import CompactTemporalDAG, mst_w_upper_bound, shortest_path
from temporal_mst_exact import exact_mst_w, incoming_weight_lower_bound
from temporal_mst_pruning import prune_dominated_events
from temporal_mst_prototype import (
    TemporalEdge,
    TemporalGraph,
    TimeStateDAG,
    exact_temporal_tree,
    shortest_path_tree_upper_bound,
    temporal_reachable,
    validate_temporal_tree,
)


def sharing_gap_graph(leaf_count: int, trunk_cost: int = 100) -> TemporalGraph:
    vertices = ["r", "hub"] + [f"leaf{position}" for position in range(leaf_count)]
    edges = [
        TemporalEdge("early_trunk", "r", "hub", 0, 1, trunk_cost),
        TemporalEdge("late_hub", "r", "hub", 10, 11, 1),
    ]
    for vertex in vertices[2:]:
        edges.extend((
            TemporalEdge(f"direct_{vertex}", "r", vertex, 0, 2, trunk_cost - 1),
            TemporalEdge(f"shared_{vertex}", "hub", vertex, 1, 2, 1),
        ))
    return TemporalGraph(vertices, edges)


def random_graph(seed: int, vertex_count: int = 5, edge_count: int = 12) -> TemporalGraph:
    rng = random.Random(seed)
    vertices = ["r"] + [f"v{position}" for position in range(1, vertex_count)]
    edges = []
    for position in range(edge_count):
        tail = rng.choice(vertices)
        head = rng.choice(vertices)
        departure = rng.randrange(-2, 10)
        edges.append(TemporalEdge(
            f"edge{position}", tail, head, departure,
            departure + rng.randrange(1, 5), rng.randrange(0, 11),
        ))
    return TemporalGraph(vertices, edges)


class ExactMSTTests(unittest.TestCase):
    def check_result(self, graph: TemporalGraph, result) -> None:
        chosen = [edge for edge in graph.edges if edge.eid in result.edge_ids]
        required = temporal_reachable(graph, "r", graph.edges)
        self.assertTrue(validate_temporal_tree(graph, "r", chosen, required))
        self.assertEqual(set(result.reachable), required)
        self.assertAlmostEqual(result.cost, sum(edge.weight for edge in chosen))

    def test_sharing_counterexample(self) -> None:
        graph = sharing_gap_graph(4)
        upper, _ = shortest_path_tree_upper_bound(TimeStateDAG(graph, "r"))
        self.assertEqual(upper, 397)
        for backend in ("explicit", "implicit"):
            result = exact_mst_w(graph, "r", backend=backend)
            self.assertEqual(result.cost, 104)
            self.check_result(graph, result)
        self.assertEqual(incoming_weight_lower_bound(graph, "r"), 5)

    def test_random_against_edge_subset_oracle(self) -> None:
        for seed in range(300):
            graph = random_graph(seed, vertex_count=2 + seed % 6, edge_count=6 + seed % 13)
            expected, _ = exact_temporal_tree(graph, "r")
            for backend in ("explicit", "channel_explicit", "implicit"):
                with self.subTest(seed=seed, backend=backend):
                    result = exact_mst_w(graph, "r", backend=backend)
                    self.assertAlmostEqual(result.cost, expected)
                    self.check_result(graph, result)
                    self.assertLessEqual(incoming_weight_lower_bound(graph, "r"), result.cost)

    def test_empty_and_disconnected(self) -> None:
        for graph in (
            TemporalGraph(["r"], []),
            TemporalGraph(["r", "unreachable"], []),
            TemporalGraph(["r", "a", "unreachable"], [TemporalEdge("edge", "r", "a", 1, 2, 3)]),
        ):
            for backend in ("explicit", "implicit"):
                self.check_result(graph, exact_mst_w(graph, "r", backend=backend))

    def test_guards(self) -> None:
        with self.assertRaisesRegex(ValueError, "terminals"):
            exact_mst_w(sharing_gap_graph(3), "r", max_terminals=2)
        with self.assertRaisesRegex(ValueError, "DP cells"):
            exact_mst_w(sharing_gap_graph(3), "r", max_dp_cells=1)
        with self.assertRaisesRegex(ValueError, "positive durations"):
            exact_mst_w(TemporalGraph(["r", "a"], [TemporalEdge("zero", "r", "a", 1, 1, 1)]), "r")

    def test_floating_weights_and_time_ties(self) -> None:
        graph = TemporalGraph(["r", "a", "b"], [
            TemporalEdge("one", "r", "a", 1, 2, 0.1),
            TemporalEdge("two", "a", "b", 2, 3, 0.2),
            TemporalEdge("three", "r", "b", 1, 3, 0.4),
            TemporalEdge("parallel", "r", "a", 1, 2, 0.5),
        ])
        for backend in ("explicit", "implicit"):
            result = exact_mst_w(graph, "r", backend=backend)
            self.assertTrue(math.isclose(result.cost, 0.3))
            self.check_result(graph, result)

    def test_compact_tree_allocates_arrays_once(self) -> None:
        graph = CompactTemporalDAG.from_arrays(
            [0, 0, 0, 0], [1, 2, 3, 4], [1, 1, 1, 1],
            [2, 2, 2, 2], [1, 2, 3, 4], vertex_count=5,
        )
        paths = shortest_path(graph, 0, return_paths=True)
        with patch("temporal_dag_framework.shortest_path", return_value=paths):
            with patch("temporal_dag_framework.np.full", wraps=np.full) as allocations:
                cost, edges, _ = mst_w_upper_bound(graph, 0)
        self.assertEqual(cost, 10)
        self.assertEqual(len(edges), 4)
        self.assertEqual(allocations.call_count, 2)

    def test_dominance_keeps_optimum(self) -> None:
        for seed in range(300):
            graph = random_graph(seed, vertex_count=2 + seed % 6, edge_count=6 + seed % 13)
            pruned, witnesses = prune_dominated_events(graph)
            by_id = {edge.eid: edge for edge in graph.edges}
            kept = {edge.eid for edge in pruned.edges}
            for dropped_id, witness_id in witnesses.items():
                dropped, witness = by_id[dropped_id], by_id[witness_id]
                self.assertIn(witness_id, kept)
                self.assertEqual((dropped.u, dropped.v), (witness.u, witness.v))
                self.assertGreaterEqual(witness.start, dropped.start)
                self.assertLessEqual(witness.arrival, dropped.arrival)
                self.assertLessEqual(witness.weight, dropped.weight)
            self.assertEqual(temporal_reachable(graph, "r", graph.edges), temporal_reachable(pruned, "r", pruned.edges))
            self.assertEqual(exact_mst_w(graph, "r").cost, exact_mst_w(pruned, "r").cost)

    def test_dominance_is_not_just_cheaper_and_earlier(self) -> None:
        graph = TemporalGraph(["r", "a", "b"], [
            TemporalEdge("root", "r", "a", 1, 3, 1),
            TemporalEdge("usable", "a", "b", 3, 5, 5),
            TemporalEdge("too_early", "a", "b", 1, 2, 1),
            TemporalEdge("dominated", "a", "b", 2, 6, 6),
            TemporalEdge("duplicate", "a", "b", 3, 5, 5),
        ])
        pruned, witnesses = prune_dominated_events(graph)
        self.assertEqual(len(witnesses), 2)
        self.assertEqual(exact_mst_w(pruned, "r").cost, 6)


if __name__ == "__main__":
    unittest.main()
