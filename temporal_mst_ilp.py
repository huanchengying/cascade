#!/usr/bin/env python3
"""Ikuta--Akiba style MILP baseline for weighted temporal spanning trees.

The formulation follows Section 3.1 of Ikuta and Akiba (NDA@SIGMOD 2016):

* ``x_e`` selects a physical temporal edge;
* one incoming edge is selected for every root-reachable non-root vertex;
* prefix variables ``y_e`` encode whether the selected incoming edge of a
  vertex arrives no later than a given event;
* temporal-feasibility constraints link every selected outgoing edge to the
  corresponding incoming prefix; and
* directed connectivity cuts are generated iteratively when zero-duration
  events can create a same-time directed cycle.

For strictly positive durations the cuts are redundant.  Indeed, the selected
parent edge of a tail must arrive before a selected outgoing edge departs, and
the outgoing edge arrives strictly later; arrival times would therefore have
to increase strictly around a directed cycle.  Since every non-root vertex has
exactly one selected incoming edge, acyclicity also implies root connectivity.

When zero-duration events are present, SciPy's HiGHS interface does not expose
lazy-constraint callbacks, so this module uses an exact outer cut loop: solve
the MILP, separate every disconnected integer component, add the violated cuts,
and solve again.  This preserves the Ikuta formulation and optimal solution,
although it is not a claim of matching their CPLEX callback engineering.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from temporal_mst_prototype import (
    TemporalEdge,
    TemporalGraph,
    earliest_arrival_tree,
    validate_temporal_tree,
)


@dataclass
class IkutaILPResult:
    cost: float
    edge_ids: List[str]
    reachable: List[str]
    optimal: bool
    status: str
    lower_bound: float
    mip_gap: float
    solve_time_s: float
    rounds: int
    cuts_added: int
    edge_variables: int
    prefix_variables: int
    constraints: int
    connectivity_mode: str
    cuts_proven_redundant: bool
    solver: str = "scipy-highs"


def _candidate_edges(
    graph: TemporalGraph,
    root: str,
) -> Tuple[List[str], List[TemporalEdge]]:
    earliest, _ = earliest_arrival_tree(graph, root)
    reachable = [vertex for vertex in graph.vertices if earliest[vertex] < math.inf]
    reachable_set = set(reachable)
    edges = [
        edge
        for edge in graph.edges
        if edge.u in reachable_set
        and edge.v in reachable_set
        and edge.v != root
        and edge.u != edge.v
        and earliest[edge.u] <= edge.start
    ]
    return reachable, edges


def _weak_components(
    vertices: Sequence[str],
    selected: Iterable[TemporalEdge],
) -> List[FrozenSet[str]]:
    adjacency = {vertex: [] for vertex in vertices}
    for edge in selected:
        adjacency[edge.u].append(edge.v)
        adjacency[edge.v].append(edge.u)
    unseen = set(vertices)
    components: List[FrozenSet[str]] = []
    while unseen:
        seed = next(iter(unseen))
        stack = [seed]
        component = set()
        unseen.remove(seed)
        while stack:
            vertex = stack.pop()
            component.add(vertex)
            for neighbor in adjacency[vertex]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(frozenset(component))
    return components


def ikuta_ilp(
    graph: TemporalGraph,
    root: str,
    *,
    time_limit: Optional[float] = None,
    mip_rel_gap: float = 0.0,
    presolve: bool = True,
    max_cut_rounds: int = 10_000,
    initial_cut_sets: Optional[Sequence[FrozenSet[str]]] = None,
) -> IkutaILPResult:
    """Solve ``MST_w`` with the Ikuta--Akiba edge/prefix formulation.

    The project convention is to span all vertices temporally reachable from
    ``root`` in the input graph.  Vertices outside that set are reported but do
    not make the optimization model infeasible.
    """

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - exercised without extras
        raise RuntimeError(
            "Ikuta ILP requires SciPy with scipy.optimize.milp; "
            "install requirements-research.txt"
        ) from exc

    if root not in graph.vertices:
        raise ValueError(f"unknown root vertex: {root}")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if mip_rel_gap < 0:
        raise ValueError("mip_rel_gap must be non-negative")
    if max_cut_rounds < 1:
        raise ValueError("max_cut_rounds must be positive")

    started = time.monotonic()

    reachable, edges = _candidate_edges(graph, root)
    required = set(reachable)
    if len(reachable) == 1:
        return IkutaILPResult(
            cost=0.0,
            edge_ids=[],
            reachable=reachable,
            optimal=True,
            status="optimal",
            lower_bound=0.0,
            mip_gap=0.0,
            solve_time_s=0.0,
            rounds=0,
            cuts_added=0,
            edge_variables=0,
            prefix_variables=0,
            constraints=0,
            connectivity_mode="trivial",
            cuts_proven_redundant=True,
        )

    edge_count = len(edges)
    incoming: Dict[str, List[int]] = {vertex: [] for vertex in reachable}
    for position, edge in enumerate(edges):
        incoming[edge.v].append(position)
    for vertex in reachable:
        if vertex != root and not incoming[vertex]:
            raise RuntimeError(f"no candidate incoming edge for reachable vertex {vertex}")
        incoming[vertex].sort(key=lambda position: (edges[position].arrival, position))
    incoming_arrivals = {
        vertex: [edges[position].arrival for position in positions]
        for vertex, positions in incoming.items()
    }
    cuts_proven_redundant = all(edge.arrival > edge.start for edge in edges)
    connectivity_mode = (
        "cut-free-positive-duration"
        if cuts_proven_redundant
        else "iterative-connectivity-cuts"
    )

    variable_count = 2 * edge_count
    objective = np.zeros(variable_count, dtype=float)
    objective[:edge_count] = [edge.weight for edge in edges]
    integrality = np.ones(variable_count, dtype=np.uint8)
    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)

    base_rows: List[Tuple[Dict[int, float], float, float]] = []

    def add_base(coefficients: Dict[int, float], lower: float, upper: float) -> None:
        base_rows.append((coefficients, lower, upper))

    for vertex in reachable:
        if vertex == root:
            continue
        add_base({position: 1.0 for position in incoming[vertex]}, 1.0, 1.0)

    for vertex in reachable:
        if vertex == root:
            continue
        previous: Optional[int] = None
        for position in incoming[vertex]:
            coefficients = {edge_count + position: 1.0, position: -1.0}
            if previous is not None:
                coefficients[edge_count + previous] = -1.0
            add_base(coefficients, 0.0, 0.0)
            previous = position

    for position, edge in enumerate(edges):
        if edge.u == root:
            continue
        prefix_offset = bisect_right(incoming_arrivals[edge.u], edge.start) - 1
        if prefix_offset < 0:
            add_base({position: 1.0}, 0.0, 0.0)
            continue
        prefix_position = incoming[edge.u][prefix_offset]
        add_base(
            {position: 1.0, edge_count + prefix_position: -1.0},
            -math.inf,
            0.0,
        )

    cut_sets: List[FrozenSet[str]] = list(dict.fromkeys(initial_cut_sets or []))
    for component in cut_sets:
        if not component or root in component or not component <= required:
            raise ValueError("initial connectivity cuts must be nonempty subsets excluding the root")
    seen_cuts = set(cut_sets)
    last_result = None

    def build_constraint() -> LinearConstraint:
        rows = list(base_rows)
        for component in cut_sets:
            coefficients = {
                position: 1.0
                for position, edge in enumerate(edges)
                if edge.v in component and edge.u not in component
            }
            if not coefficients:
                raise RuntimeError("connectivity cut has no entering candidate edge")
            rows.append((coefficients, 1.0, math.inf))
        row_indices: List[int] = []
        column_indices: List[int] = []
        values: List[float] = []
        row_lower = np.empty(len(rows), dtype=float)
        row_upper = np.empty(len(rows), dtype=float)
        for row_index, (coefficients, lower, upper) in enumerate(rows):
            row_lower[row_index] = lower
            row_upper[row_index] = upper
            for column, value in coefficients.items():
                row_indices.append(row_index)
                column_indices.append(column)
                values.append(value)
        matrix = coo_matrix(
            (values, (row_indices, column_indices)),
            shape=(len(rows), variable_count),
        ).tocsr()
        return LinearConstraint(matrix, row_lower, row_upper)

    for round_number in range(1, max_cut_rounds + 1):
        constraint = build_constraint()
        elapsed = time.monotonic() - started
        if time_limit is not None and elapsed >= time_limit:
            raise TimeoutError("Ikuta ILP exhausted its time limit before a feasible tree")
        options = {
            "presolve": presolve,
            "mip_rel_gap": mip_rel_gap,
            "disp": False,
        }
        if time_limit is not None:
            options["time_limit"] = max(1e-6, time_limit - elapsed)
        last_result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=constraint,
            options=options,
        )
        if last_result.x is None:
            if last_result.status == 1:
                raise TimeoutError("Ikuta ILP reached its time limit without an incumbent")
            raise RuntimeError(f"Ikuta ILP failed: {last_result.message}")

        selected = [
            edge for position, edge in enumerate(edges)
            if last_result.x[position] >= 0.5
        ]
        if cuts_proven_redundant:
            disconnected: List[FrozenSet[str]] = []
        else:
            components = _weak_components(reachable, selected)
            disconnected = [component for component in components if root not in component]
        if not disconnected:
            if not validate_temporal_tree(graph, root, selected, required):
                raise RuntimeError("connected ILP incumbent violates temporal-tree semantics")
            cost = math.fsum(edge.weight for edge in selected)
            lower_bound = float(getattr(last_result, "mip_dual_bound", cost))
            gap = float(getattr(last_result, "mip_gap", 0.0))
            optimal = last_result.status == 0 and gap <= 1e-9
            status = "optimal" if optimal else "feasible"
            return IkutaILPResult(
                cost=cost,
                edge_ids=sorted(edge.eid for edge in selected),
                reachable=reachable,
                optimal=optimal,
                status=status,
                lower_bound=lower_bound,
                mip_gap=gap,
                solve_time_s=time.monotonic() - started,
                rounds=round_number,
                cuts_added=len(cut_sets),
                edge_variables=edge_count,
                prefix_variables=edge_count,
                constraints=len(base_rows) + len(cut_sets),
                connectivity_mode=connectivity_mode,
                cuts_proven_redundant=cuts_proven_redundant,
            )

        new_cuts = 0
        for component in disconnected:
            if component not in seen_cuts:
                seen_cuts.add(component)
                cut_sets.append(component)
                new_cuts += 1
        if new_cuts == 0:
            raise RuntimeError("cut generation stalled on a disconnected incumbent")

    message = last_result.message if last_result is not None else "no solve attempted"
    raise RuntimeError(f"Ikuta ILP exceeded {max_cut_rounds} cut rounds: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--max-cut-rounds", type=int, default=10_000)
    args = parser.parse_args()
    graph = TemporalGraph.from_json(args.input)
    result = ikuta_ilp(
        graph,
        args.root,
        time_limit=args.time_limit,
        mip_rel_gap=args.mip_gap,
        max_cut_rounds=args.max_cut_rounds,
    )
    print(json.dumps(asdict(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
