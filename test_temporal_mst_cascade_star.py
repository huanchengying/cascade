#!/usr/bin/env python3
"""Correctness checks for batch-relaxed shared-activation stars."""

import math
import random
import unittest
from itertools import product

from temporal_mst_cascade_star import _apply_cascade_move, _best_cascade_move, cascade_activation_star
from temporal_mst_prototype import TemporalEdge, TemporalGraph, temporal_reachable, validate_temporal_tree
from temporal_mst_shared import _parent_exchange_descent, shortest_path_union_tree
from temporal_mst_star import _TreeIndex, shared_activation_star
from test_temporal_mst_exact import random_graph


def cascade_witness() -> TemporalGraph:
    edges = [
        TemporalEdge("rh-late", "r", "h", 0, 10, 1),
        TemporalEdge("rh-early", "r", "h", 0, 2, 100),
        TemporalEdge("rv", "r", "v", 0, 1, 60),
        TemporalEdge("vz", "v", "z", 5, 6, 1),
        TemporalEdge("hv", "h", "v", 2, 8, 1),
        TemporalEdge("hz", "h", "z", 2, 4, 1),
    ]
    for position in range(3):
        leaf = f"l{position}"
        edges.extend(
            [
                TemporalEdge(f"r{leaf}", "r", leaf, 0, 1, 31),
                TemporalEdge(f"h{leaf}", "h", leaf, 2, 3, 1),
            ]
        )
    return TemporalGraph(["r", "h", "v", "z", "l0", "l1", "l2"], edges)


def cascade_gap_graph(pair_count: int, trunk_cost: int = 100) -> TemporalGraph:
    vertices = ["r", "h"]
    edges = [
        TemporalEdge("rh-late", "r", "h", 0, 10, 1),
        TemporalEdge("rh-early", "r", "h", 0, 2, trunk_cost),
    ]
    for position in range(pair_count):
        parent = f"v{position}"
        child = f"z{position}"
        vertices.extend([parent, child])
        edges.extend(
            [
                TemporalEdge(
                    f"r{parent}", "r", parent, 0, 1, trunk_cost - 1
                ),
                TemporalEdge(f"{parent}{child}", parent, child, 5, 6, 1),
                TemporalEdge(f"h{parent}", "h", parent, 2, 8, 1),
                TemporalEdge(f"h{child}", "h", child, 2, 4, 1),
            ]
        )
    return TemporalGraph(vertices, edges)


class CascadeStarTests(unittest.TestCase):
    def test_unbounded_gap_from_conservative_star_local_optimum(self) -> None:
        pair_count = 8
        trunk_cost = 100
        graph = cascade_gap_graph(pair_count, trunk_cost)
        conservative = shared_activation_star(graph, "r", max_rounds=20)
        cascade = cascade_activation_star(
            graph,
            "r",
            conservative_rounds=20,
            cascade_rounds=1,
            max_hubs=None,
            max_parent_evaluations=None,
        )
        self.assertEqual(conservative.cost, 1 + pair_count * trunk_cost)
        self.assertEqual(cascade.cost, trunk_cost + 2 * pair_count)
        self.assertEqual(cascade.activated_children, 2 * pair_count)
        self.assertEqual(cascade.relaxed_deadline_vertices, pair_count)

    def test_relaxes_deadline_only_after_joint_child_replacement(self) -> None:
        graph = cascade_witness()
        conservative = shared_activation_star(graph, "r", max_rounds=4)
        cascade = cascade_activation_star(
            graph,
            "r",
            conservative_rounds=4,
            cascade_rounds=1,
            max_hubs=None,
            max_parent_evaluations=None,
        )
        self.assertEqual(conservative.cost, 155)
        self.assertEqual(cascade.exchange_cost, 155)
        self.assertEqual(cascade.cost, 105)
        self.assertEqual(cascade.accepted_moves, 1)
        self.assertEqual(cascade.activated_children, 5)
        self.assertEqual(cascade.relaxed_deadline_vertices, 1)

    def test_best_move_matches_complete_single_hub_enumeration(self) -> None:
        rng = random.Random(2026100802)
        for iteration in range(30):
            graph = random_graph(
                rng.randrange(10_000_000), vertex_count=5, edge_count=16
            )
            required = temporal_reachable(graph, "r", graph.edges)
            _, baseline, _ = shortest_path_union_tree(graph, "r")
            tree, _ = _parent_exchange_descent(graph, "r", required, baseline)
            index = _TreeIndex("r", tree, required)
            current_cost = math.fsum(edge.weight for edge in tree)
            exhaustive = current_cost
            for hub in sorted(required - {"r"}):
                parents = [
                    edge
                    for edge in graph.edges
                    if edge.v == hub
                    and edge.u in required
                    and edge.u != hub
                    and index.arrival[edge.u] <= edge.start
                    and not index.is_ancestor(hub, edge.u)
                ]
                for parent_edge in parents:
                    forbidden = set()
                    cursor = parent_edge.u
                    while True:
                        forbidden.add(cursor)
                        if cursor == "r":
                            break
                        cursor = index.parent[cursor]
                    targets = sorted(required - {"r", hub})
                    choices = []
                    for target in targets:
                        options = [index.parent_edge[target]]
                        if target not in forbidden:
                            options.extend(
                                edge
                                for edge in graph.edges
                                if edge.u == hub
                                and edge.v == target
                                and edge.start >= parent_edge.arrival
                                and edge.eid != index.parent_edge[target].eid
                            )
                        choices.append(options)
                    for selected in product(*choices):
                        parent = dict(index.parent_edge)
                        parent[hub] = parent_edge
                        parent.update(zip(targets, selected))
                        candidate = list(parent.values())
                        if validate_temporal_tree(graph, "r", candidate, required):
                            exhaustive = min(
                                exhaustive,
                                math.fsum(edge.weight for edge in candidate),
                            )
            move, _, _, _ = _best_cascade_move(
                graph,
                "r",
                required,
                tree,
                max_hubs=None,
                max_parent_evaluations=None,
            )
            if move is None:
                swept = current_cost
            else:
                swept, _ = _apply_cascade_move(
                    graph, "r", required, tree, move
                )
            with self.subTest(iteration=iteration):
                self.assertAlmostEqual(swept, exhaustive)

    def test_bounded_search_is_feasible_and_monotone(self) -> None:
        rng = random.Random(2026100803)
        for iteration in range(40):
            graph = random_graph(
                rng.randrange(10_000_000), vertex_count=7, edge_count=24
            )
            result = cascade_activation_star(
                graph,
                "r",
                conservative_rounds=2,
                cascade_rounds=2,
                max_hubs=4,
                max_parent_evaluations=8,
            )
            selected = [edge for edge in graph.edges if edge.eid in result.edge_ids]
            required = temporal_reachable(graph, "r", graph.edges)
            with self.subTest(iteration=iteration):
                self.assertLessEqual(result.cost, result.conservative_star_cost + 1e-7)
                self.assertTrue(validate_temporal_tree(graph, "r", selected, required))


if __name__ == "__main__":
    unittest.main()
