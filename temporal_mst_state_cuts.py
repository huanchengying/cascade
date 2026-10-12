#!/usr/bin/env python3
"""Projected time-state connectivity cuts for the compact prefix LP.

The explicit temporal-state graph has one arrival state per distinct incoming
arrival, fixed free waiting arcs, and one paid arc per temporal event.  A
source-to-state cut that does not cross a fixed waiting arc is a per-vertex
arrival-prefix set.  Projecting its paid crossing arcs gives

    sum(x_e for paid state arcs entering Q) >= y_v(t).

Uniform deadline-prefix cuts are the special case in which every physical
vertex on the sink side uses the same threshold t.  This module exists to
audit that relationship and LP strength; the underlying state-cut family is
standard Directed Steiner machinery and is not claimed as novel.
"""

from __future__ import annotations

import math
import time
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

from temporal_mst_certificate import ConnectivityCut, PrefixLPResult, PrefixModel, _cut_matrix, cut_row


State = Tuple[str, int]
SOURCE: State = ("__root_source__", -1)


@dataclass(frozen=True)
class StatePrefixCut:
    thresholds: Tuple[Tuple[str, float], ...]
    target: str
    deadline: float


@dataclass
class StatePrefixLPResult:
    lower_bound: float
    lp_objective: float
    edge_excess: np.ndarray
    fractions: np.ndarray
    cuts: List[StatePrefixCut]
    rounds: int
    status: str
    dual_correction: float
    max_primal_violation: float
    wall_time_s: float


class StateExpansion:
    def __init__(self, model: PrefixModel):
        self.model = model
        self.times: Dict[str, Tuple[float, ...]] = {
            vertex: tuple(sorted(set(arrivals)))
            for vertex, arrivals in model.arrivals.items()
        }
        self.paid_arcs: List[Tuple[State, State]] = []
        for edge in model.edges:
            head_times = self.times[edge.v]
            head_index = bisect_right(head_times, edge.arrival) - 1
            if head_index < 0 or head_times[head_index] != edge.arrival:
                raise RuntimeError("event head state is missing")
            head = (edge.v, head_index)
            if edge.u == model.root:
                tail = SOURCE
            else:
                tail_times = self.times[edge.u]
                tail_index = bisect_right(tail_times, edge.start) - 1
                if tail_index < 0:
                    raise RuntimeError("candidate event has no reachable tail state")
                tail = (edge.u, tail_index)
            self.paid_arcs.append((tail, head))
        self.waiting_arcs = [
            ((vertex, index), (vertex, index + 1))
            for vertex, times in self.times.items()
            for index in range(len(times) - 1)
        ]

    def target_state(self, vertex: str, deadline: float) -> State:
        index = bisect_right(self.times[vertex], deadline) - 1
        if index < 0:
            raise ValueError("target deadline precedes every arrival state")
        return vertex, index

    def prefix_variable(self, vertex: str, deadline: float) -> int:
        position = bisect_right(self.model.arrivals[vertex], deadline) - 1
        if position < 0:
            raise ValueError("target deadline has no compact prefix variable")
        return len(self.model.edges) + self.model.incoming[vertex][position]

    def thresholds_from_side(self, side: set) -> Tuple[Tuple[str, float], ...]:
        thresholds = []
        for vertex, times in self.times.items():
            selected = [index for index in range(len(times)) if (vertex, index) in side]
            if not selected:
                continue
            last = max(selected)
            if selected != list(range(last + 1)):
                raise RuntimeError("state cut crosses a fixed waiting arc")
            thresholds.append((vertex, times[last]))
        return tuple(sorted(thresholds))

    def crossing_positions(self, cut: StatePrefixCut) -> List[int]:
        threshold = dict(cut.thresholds)
        crossing = []
        for position, (tail, head) in enumerate(self.paid_arcs):
            head_vertex, head_index = head
            head_inside = self.times[head_vertex][head_index] <= threshold.get(
                head_vertex, -math.inf
            )
            if not head_inside:
                continue
            if tail == SOURCE:
                tail_inside = False
            else:
                tail_vertex, tail_index = tail
                tail_inside = self.times[tail_vertex][tail_index] <= threshold.get(
                    tail_vertex, -math.inf
                )
            if not tail_inside:
                crossing.append(position)
        return crossing


def state_cut_row(
    expansion: StateExpansion,
    cut: StatePrefixCut,
) -> Tuple[Dict[int, float], float]:
    threshold = dict(cut.thresholds)
    if expansion.model.root in threshold:
        raise ValueError("state-prefix cut cannot contain the root source")
    if threshold.get(cut.target, -math.inf) < cut.deadline:
        raise ValueError("state-prefix cut must contain its target state")
    coefficients = {
        position: -1.0 for position in expansion.crossing_positions(cut)
    }
    coefficients[expansion.prefix_variable(cut.target, cut.deadline)] = 1.0
    return coefficients, 0.0


def _matrix(
    model: PrefixModel,
    expansion: StateExpansion,
    cuts: Sequence[StatePrefixCut],
) -> Tuple[csr_matrix, np.ndarray]:
    rows: List[int] = []
    columns: List[int] = []
    values: List[float] = []
    for row, cut in enumerate(cuts):
        coefficients, _ = state_cut_row(expansion, cut)
        for column, value in coefficients.items():
            rows.append(row)
            columns.append(column)
            values.append(value)
    cut_matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(cuts), len(model.objective)),
    ).tocsr()
    return (
        vstack((model.support, cut_matrix), format="csr"),
        np.zeros(model.support.shape[0] + len(cuts)),
    )


def _queries(
    model: PrefixModel,
    fractions: np.ndarray,
    *,
    max_probes: int,
    tolerance: float,
    all_prefixes: bool,
) -> List[Tuple[str, float, float]]:
    edge_count = len(model.edges)
    if all_prefixes:
        result = []
        for vertex in model.reachable:
            if vertex == model.root:
                continue
            for deadline in sorted(set(model.arrivals[vertex])):
                prefix = bisect_right(model.arrivals[vertex], deadline) - 1
                mass = float(
                    fractions[edge_count + model.incoming[vertex][prefix]]
                )
                if mass > tolerance:
                    result.append((vertex, deadline, mass))
        return result
    result: List[Tuple[str, float, float]] = []
    ranked = sorted(
        (vertex for vertex in model.reachable if vertex != model.root),
        key=lambda vertex: (-len(model.incoming[vertex]), vertex),
    )
    for vertex in ranked[:max_probes]:
        deadline = model.arrivals[vertex][-1]
        variable = edge_count + model.incoming[vertex][-1]
        result.append((vertex, deadline, float(fractions[variable])))

    conditional: Dict[Tuple[str, float], float] = {}
    for edge in model.edges:
        if edge.u == model.root:
            continue
        prefix = bisect_right(model.arrivals[edge.u], edge.start) - 1
        if prefix < 0:
            continue
        deadline = model.arrivals[edge.u][prefix]
        mass = float(fractions[edge_count + model.incoming[edge.u][prefix]])
        if mass > tolerance:
            key = edge.u, deadline
            conditional[key] = max(conditional.get(key, 0.0), mass)
    for (vertex, deadline), mass in sorted(
        conditional.items(), key=lambda item: (-item[1], item[0])
    )[:max_probes]:
        result.append((vertex, deadline, mass))
    return list(dict.fromkeys(result))


def separate_state_prefix_cuts(
    model: PrefixModel,
    expansion: StateExpansion,
    fractions: np.ndarray,
    *,
    max_probes: int = 24,
    tolerance: float = 1e-7,
    all_prefixes: bool = False,
) -> List[StatePrefixCut]:
    import networkx as nx

    network = nx.DiGraph()
    network.add_node(SOURCE)
    for vertex, times in expansion.times.items():
        network.add_nodes_from((vertex, index) for index in range(len(times)))
    for tail, head in expansion.waiting_arcs:
        network.add_edge(tail, head, capacity=1.0, fixed_wait=True)
    for position, (tail, head) in enumerate(expansion.paid_arcs):
        capacity = max(0.0, float(fractions[position]))
        if network.has_edge(tail, head):
            network[tail][head]["capacity"] += capacity
            network[tail][head].setdefault("paid_positions", []).append(position)
        else:
            network.add_edge(
                tail,
                head,
                capacity=capacity,
                paid_positions=[position],
            )

    cuts = []
    seen = set()
    for target, deadline, demand in _queries(
        model,
        fractions,
        max_probes=max_probes,
        tolerance=tolerance,
        all_prefixes=all_prefixes,
    ):
        if demand <= tolerance:
            continue
        target_state = expansion.target_state(target, deadline)
        capacity, partition = nx.minimum_cut(
            network,
            SOURCE,
            target_state,
            flow_func=nx.algorithms.flow.preflow_push,
        )
        if capacity >= demand - tolerance:
            continue
        sink_side = set(partition[1])
        if any(tail not in sink_side and head in sink_side for tail, head in expansion.waiting_arcs):
            raise RuntimeError("violated state cut unexpectedly crosses a fixed waiting arc")
        cut = StatePrefixCut(
            expansion.thresholds_from_side(sink_side), target, deadline
        )
        coefficients, bound = state_cut_row(expansion, cut)
        violation = (
            math.fsum(
                value * float(fractions[position])
                for position, value in coefficients.items()
            )
            - bound
        )
        if violation > tolerance and cut not in seen:
            cuts.append(cut)
            seen.add(cut)
    return cuts


def solve_state_prefix_lp(
    model: PrefixModel,
    *,
    max_rounds: int = 4,
    max_probes: int = 24,
    time_limit: Optional[float] = 30.0,
    all_prefixes: bool = False,
) -> StatePrefixLPResult:
    if max_rounds < 1 or max_probes < 1:
        raise ValueError("invalid state-cut separation settings")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if not model.edges:
        return StatePrefixLPResult(
            0.0,
            0.0,
            np.array([]),
            np.array([]),
            [],
            0,
            "trivial",
            0.0,
            0.0,
            0.0,
        )

    started = time.monotonic()
    expansion = StateExpansion(model)
    cuts: List[StatePrefixCut] = []
    seen = set()
    result = None
    used_cuts: List[StatePrefixCut] = []
    round_number = 0
    for round_number in range(1, max_rounds + 1):
        remaining = None if time_limit is None else time_limit - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            break
        matrix, rhs = _matrix(model, expansion, cuts)
        options = {} if remaining is None else {"time_limit": remaining}
        attempt = linprog(
            model.objective,
            A_ub=matrix,
            b_ub=rhs,
            A_eq=model.equality,
            b_eq=model.rhs,
            bounds=(0.0, None),
            method="highs",
            options=options,
        )
        if attempt.status != 0:
            if result is None:
                raise TimeoutError(f"state-prefix LP did not finish: {attempt.message}")
            break
        result = attempt
        used_cuts = list(cuts)
        if round_number == max_rounds:
            break
        found = separate_state_prefix_cuts(
            model,
            expansion,
            result.x,
            max_probes=max_probes,
            all_prefixes=all_prefixes,
        )
        new_cuts = [cut for cut in found if cut not in seen]
        if not new_cuts:
            break
        cuts.extend(new_cuts)
        seen.update(new_cuts)
    if result is None:
        raise TimeoutError("state-prefix LP budget expired before its first solve")

    matrix, rhs = _matrix(model, expansion, used_cuts)
    equality_prices = np.asarray(result.eqlin.marginals)
    inequality_prices = np.minimum(0.0, np.asarray(result.ineqlin.marginals))
    residual = model.objective - model.equality.T @ equality_prices - matrix.T @ inequality_prices
    if not (
        np.all(np.isfinite(equality_prices))
        and np.all(np.isfinite(inequality_prices))
        and np.all(np.isfinite(residual))
    ):
        raise RuntimeError("non-finite state-prefix LP prices")
    dual_constant = math.fsum(
        float(value) * float(price)
        for value, price in zip(model.rhs, equality_prices)
    ) + math.fsum(float(value) * float(price) for value, price in zip(rhs, inequality_prices))
    correction = math.fsum(min(0.0, float(value)) for value in residual)
    arithmetic_margin = 1e-9 * max(
        1.0, float(np.sum(np.abs(model.objective))), abs(dual_constant)
    )
    lower = dual_constant + correction - arithmetic_margin
    primal_violation = max(
        0.0,
        float(np.max(np.abs(model.equality @ result.x - model.rhs), initial=0)),
        float(np.max(matrix @ result.x - rhs, initial=0)),
    )
    if primal_violation > 1e-5:
        raise RuntimeError("state-prefix LP primal residual is too large")
    return StatePrefixLPResult(
        max(0.0, lower),
        float(result.fun),
        np.maximum(0.0, residual[: len(model.edges)]),
        result.x[: len(model.edges)],
        used_cuts,
        round_number,
        "optimal-state-relaxation",
        correction,
        primal_violation,
        time.monotonic() - started,
    )


def separate_all_uniform_deadline_cuts(
    model: PrefixModel,
    fractions: np.ndarray,
    *,
    tolerance: float = 1e-7,
) -> List[ConnectivityCut]:
    """Completely separate the uniform-deadline family on small audit graphs."""

    import networkx as nx

    edge_count = len(model.edges)
    result = []
    seen = set()
    for target in model.reachable:
        if target == model.root:
            continue
        for deadline in sorted(set(model.arrivals[target])):
            prefix = bisect_right(model.arrivals[target], deadline) - 1
            demand = float(
                fractions[edge_count + model.incoming[target][prefix]]
            )
            if demand <= tolerance:
                continue
            network = nx.DiGraph()
            network.add_nodes_from(model.reachable)
            for position, edge in enumerate(model.edges):
                if edge.arrival > deadline:
                    continue
                capacity = max(0.0, float(fractions[position]))
                if network.has_edge(edge.u, edge.v):
                    network[edge.u][edge.v]["capacity"] += capacity
                else:
                    network.add_edge(edge.u, edge.v, capacity=capacity)
            capacity, partition = nx.minimum_cut(
                network,
                model.root,
                target,
                flow_func=nx.algorithms.flow.preflow_push,
            )
            if capacity >= demand - tolerance:
                continue
            cut = ConnectivityCut(frozenset(partition[1]), target, deadline)
            coefficients, bound = cut_row(model, cut)
            violation = (
                math.fsum(
                    value * float(fractions[position])
                    for position, value in coefficients.items()
                )
                - bound
            )
            if violation > tolerance and cut not in seen:
                result.append(cut)
                seen.add(cut)
    return result


def solve_complete_uniform_deadline_lp(
    model: PrefixModel,
    *,
    max_rounds: int = 20,
    time_limit: Optional[float] = 30.0,
) -> PrefixLPResult:
    """Solve the compact LP with exhaustive uniform-prefix separation."""

    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    if not model.edges:
        return PrefixLPResult(
            0.0,
            0.0,
            np.array([]),
            np.array([]),
            [],
            0,
            "trivial",
            0.0,
            0.0,
            0.0,
        )
    started = time.monotonic()
    cuts: List[ConnectivityCut] = []
    seen = set()
    result = None
    used_cuts: List[ConnectivityCut] = []
    round_number = 0
    for round_number in range(1, max_rounds + 1):
        remaining = None if time_limit is None else time_limit - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            break
        matrix, rhs = _cut_matrix(model, cuts)
        options = {} if remaining is None else {"time_limit": remaining}
        attempt = linprog(
            model.objective,
            A_ub=matrix,
            b_ub=rhs,
            A_eq=model.equality,
            b_eq=model.rhs,
            bounds=(0.0, None),
            method="highs",
            options=options,
        )
        if attempt.status != 0:
            if result is None:
                raise TimeoutError(f"complete deadline LP did not finish: {attempt.message}")
            break
        result = attempt
        used_cuts = list(cuts)
        if round_number == max_rounds:
            break
        found = separate_all_uniform_deadline_cuts(model, result.x)
        new_cuts = [cut for cut in found if cut not in seen]
        if not new_cuts:
            break
        cuts.extend(new_cuts)
        seen.update(new_cuts)
    if result is None:
        raise TimeoutError("complete deadline LP budget expired before its first solve")

    matrix, rhs = _cut_matrix(model, used_cuts)
    equality_prices = np.asarray(result.eqlin.marginals)
    inequality_prices = np.minimum(0.0, np.asarray(result.ineqlin.marginals))
    residual = model.objective - model.equality.T @ equality_prices - matrix.T @ inequality_prices
    dual_constant = math.fsum(
        float(value) * float(price)
        for value, price in zip(model.rhs, equality_prices)
    ) + math.fsum(float(value) * float(price) for value, price in zip(rhs, inequality_prices))
    correction = math.fsum(min(0.0, float(value)) for value in residual)
    arithmetic_margin = 1e-9 * max(
        1.0, float(np.sum(np.abs(model.objective))), abs(dual_constant)
    )
    primal_violation = max(
        0.0,
        float(np.max(np.abs(model.equality @ result.x - model.rhs), initial=0)),
        float(np.max(matrix @ result.x - rhs, initial=0)),
    )
    return PrefixLPResult(
        max(0.0, dual_constant + correction - arithmetic_margin),
        float(result.fun),
        np.maximum(0.0, residual[: len(model.edges)]),
        result.x[: len(model.edges)],
        used_cuts,
        round_number,
        "optimal-complete-uniform-deadline-relaxation",
        correction,
        primal_violation,
        time.monotonic() - started,
    )
