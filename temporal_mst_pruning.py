#!/usr/bin/env python3
"""Safe local event dominance; a preprocessing baseline, not a novelty claim."""

from bisect import bisect_left
from itertools import groupby
from typing import Dict, Tuple

from temporal_mst_prototype import TemporalGraph


def prune_dominated_events(graph: TemporalGraph) -> Tuple[TemporalGraph, Dict[str, str]]:
    """Remove e only with a retained same-endpoint witness f satisfying
    start(f) >= start(e), arrival(f) <= arrival(e), weight(f) <= weight(e).

    A departure-descending sweep plus prefix-min Fenwick tree takes O(m log m)
    time and O(m) space. Identical events retain the lexicographically first ID.
    """
    ordered = sorted(
        graph.edges,
        key=lambda edge: (edge.u, edge.v, -edge.start, edge.arrival, edge.weight, edge.eid),
    )
    witnesses = {}
    for _, group in groupby(ordered, key=lambda edge: (edge.u, edge.v)):
        edges = list(group)
        arrivals = sorted({edge.arrival for edge in edges})
        prefix_minimum = [None] * (len(arrivals) + 1)
        for edge in edges:
            position = bisect_left(arrivals, edge.arrival) + 1
            cursor = position
            best = None
            while cursor > 0:
                candidate = prefix_minimum[cursor]
                if candidate is not None and (best is None or candidate < best):
                    best = candidate
                cursor -= cursor & -cursor
            if best is not None and best[0] <= edge.weight:
                witnesses[edge.eid] = best[1]
                continue
            candidate = (edge.weight, edge.eid)
            cursor = position
            while cursor < len(prefix_minimum):
                if prefix_minimum[cursor] is None or candidate < prefix_minimum[cursor]:
                    prefix_minimum[cursor] = candidate
                cursor += cursor & -cursor
    return TemporalGraph(
        list(graph.vertices), [edge for edge in graph.edges if edge.eid not in witnesses]
    ), witnesses
