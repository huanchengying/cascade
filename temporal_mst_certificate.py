#!/usr/bin/env python3
"""LP pricing, deadline connectivity cuts and certified support-closed masters.

Connectivity cuts and reduced-cost fixing are established techniques. Deadline
prefix cuts are a research candidate, not an established novelty claim. Every
bound refers to the original graph; restricted-master bounds are not promoted
unless the master contains the entire safely retained event set.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

from temporal_mst_ilp import _candidate_edges, ikuta_ilp
from temporal_mst_prototype import TemporalEdge, TemporalGraph, earliest_arrival_tree, validate_temporal_tree
from temporal_mst_shared import shortest_path_union_tree


@dataclass(frozen=True)
class ConnectivityCut:
    side: FrozenSet[str]
    target: Optional[str] = None
    deadline: Optional[float] = None


@dataclass
class PrefixModel:
    graph: TemporalGraph
    root: str
    reachable: List[str]
    edges: List[TemporalEdge]
    incoming: Dict[str, List[int]]
    arrivals: Dict[str, List[float]]
    objective: np.ndarray
    equality: csr_matrix
    rhs: np.ndarray
    support: csr_matrix


@dataclass
class PrefixLPResult:
    lower_bound: float
    lp_objective: float
    edge_excess: np.ndarray
    fractions: np.ndarray
    cuts: List[ConnectivityCut]
    rounds: int
    status: str
    dual_correction: float
    max_primal_violation: float
    wall_time_s: float


@dataclass
class CertifiedPoolResult:
    cost: float
    edge_ids: List[str]
    lower_bound: float
    relative_gap: float
    certified_optimal: bool
    baseline_cost: float
    reachable: List[str]
    original_edges: int
    safe_edges: int
    pool_edges: int
    root_cuts: int
    deadline_cuts: int
    lp_objective: float
    lp_status: str
    lp_time_s: float
    total_time_s: float
    deadline_exceeded: bool
    master_trace: List[dict] = field(default_factory=list)
    algorithm: str = "deadline-priced-support-closure"


def build_prefix_model(graph: TemporalGraph, root: str) -> PrefixModel:
    reachable, edges = _candidate_edges(graph, root)
    incoming = {vertex: [] for vertex in reachable}
    for position, edge in enumerate(edges):
        incoming[edge.v].append(position)
    for positions in incoming.values():
        positions.sort(key=lambda position: (edges[position].arrival, position))
    arrivals = {vertex: [edges[position].arrival for position in positions] for vertex, positions in incoming.items()}
    edge_count = len(edges)
    rows, columns, values, rhs = [], [], [], []
    for vertex in reachable:
        if vertex == root:
            continue
        row = len(rhs)
        positions = incoming[vertex]
        rows.extend([row] * len(positions))
        columns.extend(positions)
        values.extend([1.0] * len(positions))
        rhs.append(1.0)
    for vertex in reachable:
        previous = None
        for position in incoming[vertex]:
            row = len(rhs)
            rows.extend((row, row))
            columns.extend((edge_count + position, position))
            values.extend((1.0, -1.0))
            if previous is not None:
                rows.append(row)
                columns.append(edge_count + previous)
                values.append(-1.0)
            rhs.append(0.0)
            previous = position
    equality = coo_matrix((values, (rows, columns)), shape=(len(rhs), 2 * edge_count)).tocsr()
    rows, columns, values = [], [], []
    row_count = 0
    for position, edge in enumerate(edges):
        if edge.u == root:
            continue
        prefix = bisect_right(arrivals[edge.u], edge.start) - 1
        rows.append(row_count)
        columns.append(position)
        values.append(1.0)
        if prefix >= 0:
            rows.append(row_count)
            columns.append(edge_count + incoming[edge.u][prefix])
            values.append(-1.0)
        row_count += 1
    support = coo_matrix((values, (rows, columns)), shape=(row_count, 2 * edge_count)).tocsr()
    objective = np.zeros(2 * edge_count)
    objective[:edge_count] = [edge.weight for edge in edges]
    return PrefixModel(graph, root, reachable, edges, incoming, arrivals, objective, equality, np.array(rhs), support)


def cut_row(model: PrefixModel, cut: ConnectivityCut) -> Tuple[Dict[int, float], float]:
    if not cut.side or model.root in cut.side:
        raise ValueError("a connectivity cut must exclude the root and have a nonempty side")
    coefficients = {
        position: -1.0 for position, edge in enumerate(model.edges)
        if edge.v in cut.side and edge.u not in cut.side
        and (cut.deadline is None or edge.arrival <= cut.deadline)
    }
    if cut.target is None:
        if cut.deadline is not None:
            raise ValueError("a deadline cut needs a target")
        return coefficients, -1.0
    if cut.target not in cut.side or cut.deadline is None:
        raise ValueError("a deadline target must be inside its cut side")
    prefix = bisect_right(model.arrivals[cut.target], cut.deadline) - 1
    if prefix >= 0:
        coefficients[len(model.edges) + model.incoming[cut.target][prefix]] = 1.0
    return coefficients, 0.0


def _cut_matrix(model: PrefixModel, cuts: Sequence[ConnectivityCut]) -> Tuple[csr_matrix, np.ndarray]:
    rows, columns, values, rhs = [], [], [], []
    for row, cut in enumerate(cuts):
        coefficients, bound = cut_row(model, cut)
        for column, value in coefficients.items():
            rows.append(row)
            columns.append(column)
            values.append(value)
        rhs.append(bound)
    matrix = coo_matrix((values, (rows, columns)), shape=(len(cuts), len(model.objective))).tocsr()
    return vstack((model.support, matrix), format="csr"), np.concatenate((np.zeros(model.support.shape[0]), rhs))


def separate_connectivity(
    model: PrefixModel,
    fractions: np.ndarray,
    *,
    deadline_cuts: bool,
    max_probes: int = 24,
    tolerance: float = 1e-7,
) -> List[ConnectivityCut]:
    import networkx as nx

    edge_count = len(model.edges)
    active = [(edge, max(0.0, float(fractions[position]))) for position, edge in enumerate(model.edges) if fractions[position] > 1e-10]
    candidates = []
    ranked = sorted((vertex for vertex in model.reachable if vertex != model.root), key=lambda vertex: (-len(model.incoming[vertex]), vertex))
    for vertex in ranked[:max_probes]:
        candidates.append((vertex, None, 1.0))
    if deadline_cuts:
        queries = {}
        for edge, fraction in active:
            if edge.u == model.root:
                continue
            prefix = bisect_right(model.arrivals[edge.u], edge.start) - 1
            if prefix < 0:
                continue
            mass = float(fractions[edge_count + model.incoming[edge.u][prefix]])
            if mass > tolerance:
                key = (edge.u, model.arrivals[edge.u][prefix])
                queries[key] = max(queries.get(key, 0.0), mass)
        for (vertex, deadline), mass in sorted(queries.items(), key=lambda item: (-item[1], item[0]))[:max_probes]:
            candidates.append((vertex, deadline, mass))
    result = []
    seen = set()
    for target, deadline, demand in candidates:
        network = nx.DiGraph()
        network.add_nodes_from(model.reachable)
        for edge, capacity in active:
            if deadline is not None and edge.arrival > deadline:
                continue
            if network.has_edge(edge.u, edge.v):
                network[edge.u][edge.v]["capacity"] += capacity
            else:
                network.add_edge(edge.u, edge.v, capacity=capacity)
        capacity, partition = nx.minimum_cut(network, model.root, target, flow_func=nx.algorithms.flow.preflow_push)
        if capacity >= demand - tolerance:
            continue
        cut = ConnectivityCut(frozenset(partition[1]), target if deadline is not None else None, deadline)
        coefficients, bound = cut_row(model, cut)
        violation = math.fsum(value * float(fractions[position]) for position, value in coefficients.items()) - bound
        if violation > tolerance and cut not in seen:
            result.append(cut)
            seen.add(cut)
    return result


def solve_prefix_lp(
    model: PrefixModel,
    *,
    cut_mode: str = "deadline",
    max_rounds: int = 4,
    max_probes: int = 24,
    time_limit: Optional[float] = 30.0,
) -> PrefixLPResult:
    if cut_mode not in {"none", "root", "deadline"} or max_rounds < 1 or max_probes < 1:
        raise ValueError("invalid LP separation settings")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if not model.edges:
        return PrefixLPResult(0.0, 0.0, np.array([]), np.array([]), [], 0, "trivial", 0.0, 0.0, 0.0)
    started = time.monotonic()
    cuts = []
    seen = set()
    result = None
    used_cuts = []
    for round_number in range(1, max_rounds + 1):
        remaining = None if time_limit is None else time_limit - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            break
        matrix, rhs = _cut_matrix(model, cuts)
        options = {} if remaining is None else {"time_limit": remaining}
        attempt = linprog(model.objective, A_ub=matrix, b_ub=rhs, A_eq=model.equality, b_eq=model.rhs, bounds=(0.0, None), method="highs", options=options)
        if attempt.status != 0:
            if result is None:
                raise TimeoutError(f"prefix LP did not finish: {attempt.message}")
            break
        result = attempt
        used_cuts = list(cuts)
        if cut_mode == "none" or round_number == max_rounds:
            break
        found = separate_connectivity(model, result.x, deadline_cuts=cut_mode == "deadline", max_probes=max_probes)
        new_cuts = [cut for cut in found if cut not in seen]
        if not new_cuts:
            break
        cuts.extend(new_cuts)
        seen.update(new_cuts)
    if result is None:
        raise TimeoutError("prefix LP budget expired before its first solve")
    matrix, rhs = _cut_matrix(model, used_cuts)
    equality_prices = np.array(result.eqlin.marginals)
    inequality_prices = np.minimum(0.0, result.ineqlin.marginals)
    residual = model.objective - model.equality.T @ equality_prices - matrix.T @ inequality_prices
    if not np.all(np.isfinite(equality_prices)) or not np.all(np.isfinite(inequality_prices)) or not np.all(np.isfinite(residual)):
        raise RuntimeError("non-finite LP dual prices")
    dual_constant = math.fsum(float(value) * float(price) for value, price in zip(model.rhs, equality_prices)) + math.fsum(float(value) * float(price) for value, price in zip(rhs, inequality_prices))
    correction = math.fsum(min(0.0, float(value)) for value in residual)
    arithmetic_margin = 1e-9 * max(1.0, float(np.sum(np.abs(model.objective))), abs(dual_constant))
    lower = dual_constant + correction - arithmetic_margin
    primal_violation = max(0.0, float(np.max(np.abs(model.equality @ result.x - model.rhs), initial=0)), float(np.max(matrix @ result.x - rhs, initial=0)))
    if primal_violation > 1e-5:
        raise RuntimeError("LP primal residual too large for pricing")
    return PrefixLPResult(max(0.0, lower), float(result.fun), np.maximum(0.0, residual[:len(model.edges)]), result.x[:len(model.edges)], used_cuts, round_number, "optimal-relaxation", correction, primal_violation, time.monotonic() - started)


def safe_event_ids(model: PrefixModel, lp: PrefixLPResult, upper_bound: float, incumbent_ids: Sequence[str]) -> set:
    margin = 1e-8 * max(1.0, upper_bound, abs(lp.lower_bound))
    if lp.lower_bound > upper_bound + margin:
        raise RuntimeError("LP certificate exceeds validated incumbent")
    budget = max(0.0, upper_bound - lp.lower_bound)
    result = {edge.eid for position, edge in enumerate(model.edges) if lp.edge_excess[position] <= budget + margin}
    result.update(incumbent_ids)
    return result


def support_closed_pool(model: PrefixModel, lp: PrefixLPResult, safe_ids: set, incumbent_ids: Sequence[str], width: int) -> set:
    if width < 1:
        raise ValueError("pool width must be positive")
    pool = set(incumbent_ids)
    safe_graph = TemporalGraph(list(model.graph.vertices), [edge for edge in model.edges if edge.eid in safe_ids])
    earliest, _ = earliest_arrival_tree(safe_graph, model.root)
    usable_ids = {edge.eid for edge in safe_graph.edges if earliest[edge.u] <= edge.start}
    position_by_id = {edge.eid: position for position, edge in enumerate(model.edges)}
    prefix_best = {}
    for vertex, positions in model.incoming.items():
        best = None
        best_list = []
        eligible = []
        for position in positions:
            edge = model.edges[position]
            if edge.eid in usable_ids:
                eligible.append(position)
                if best is None or (lp.edge_excess[position], edge.weight, edge.arrival, position) < (lp.edge_excess[best], model.edges[best].weight, model.edges[best].arrival, best):
                    best = position
            best_list.append(best)
        prefix_best[vertex] = best_list
        ranked = sorted(eligible, key=lambda position: (lp.edge_excess[position], model.edges[position].weight, model.edges[position].arrival, position))
        pool.update(model.edges[position].eid for position in ranked[:width])
    pool.update(edge.eid for position, edge in enumerate(model.edges) if edge.eid in usable_ids and lp.fractions[position] > 1e-8)
    pending = list(sorted(pool))
    while pending:
        edge = model.edges[position_by_id[pending.pop()]]
        if edge.u == model.root:
            continue
        prefix = bisect_right(model.arrivals[edge.u], edge.start) - 1
        support = prefix_best[edge.u][prefix] if prefix >= 0 else None
        if support is None:
            raise RuntimeError("a safely retained pool event has no safe temporal support")
        support_id = model.edges[support].eid
        if support_id not in pool:
            pool.add(support_id)
            pending.append(support_id)
    return pool


def certified_pool_optimize(
    graph: TemporalGraph,
    root: str,
    *,
    time_limit: float = 60.0,
    cut_mode: str = "deadline",
    max_lp_rounds: int = 4,
    max_probes: int = 24,
    widths: Sequence[int] = (1, 2, 4),
    safe_master_limit: int = 200_000,
) -> CertifiedPoolResult:
    if time_limit <= 0 or not widths or min(widths) < 1 or safe_master_limit < 0:
        raise ValueError("invalid certified optimizer settings")
    if any(edge.arrival <= edge.start for edge in graph.edges):
        raise ValueError("certified closure optimizer requires strictly positive durations")
    started = time.monotonic()
    model = build_prefix_model(graph, root)
    baseline, tree, _ = shortest_path_union_tree(graph, root)
    best_ids = sorted(edge.eid for edge in tree)
    best_cost = baseline
    lp_started = time.monotonic()
    lp_status = "not-run"
    trace = []
    fallback = math.fsum(min(model.edges[position].weight for position in model.incoming[vertex]) for vertex in model.reachable if vertex != root)
    try:
        remaining = time_limit - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("budget exhausted during input preparation")
        lp = solve_prefix_lp(model, cut_mode=cut_mode, max_rounds=max_lp_rounds, max_probes=max_probes, time_limit=min(0.45 * time_limit, remaining))
        lower = lp.lower_bound
        lp_status = lp.status
    except TimeoutError as exc:
        lp = PrefixLPResult(fallback, fallback, np.zeros(len(model.edges)), np.zeros(len(model.edges)), [], 0, "timeout-fallback", 0.0, 0.0, time.monotonic() - lp_started)
        lower = fallback
        lp_status = f"timeout-fallback: {exc}"
    safe_ids = safe_event_ids(model, lp, best_cost, best_ids)
    last_pool = set(best_ids)
    certified = best_cost - lower <= 1e-7 * max(1.0, best_cost)
    settings = list(widths) + [None]
    for stage, width in enumerate(settings):
        remaining = time_limit - (time.monotonic() - started)
        if certified or remaining <= 0:
            break
        safe_ids = safe_event_ids(model, lp, best_cost, best_ids)
        if width is None:
            if len(safe_ids) > safe_master_limit:
                continue
            pool = set(safe_ids)
        else:
            pool = support_closed_pool(model, lp, safe_ids, best_ids, width)
        complete = pool == safe_ids
        last_pool = pool
        master_graph = TemporalGraph(list(graph.vertices), [edge for edge in model.edges if edge.eid in pool])
        allowance = remaining if complete or width is None else remaining / max(1, len(settings) - stage)
        master_started = time.monotonic()
        record = {"width": width, "pool_edges": len(pool), "safe_edges": len(safe_ids), "globally_equivalent": complete, "solver_allowance_s": allowance}
        try:
            result = ikuta_ilp(master_graph, root, time_limit=allowance)
            chosen = [edge for edge in model.edges if edge.eid in set(result.edge_ids)]
            if not validate_temporal_tree(graph, root, chosen, set(model.reachable)):
                raise RuntimeError("restricted master omitted a required vertex")
            if result.cost < best_cost:
                best_cost, best_ids = result.cost, sorted(result.edge_ids)
            if complete:
                lower = max(lower, result.lower_bound)
                certified = result.optimal and result.mip_gap <= 1e-7
            certified = certified or best_cost - lower <= 1e-7 * max(1.0, best_cost)
            record.update(status=result.status, cost=result.cost, restricted_lower_bound=result.lower_bound, mip_gap=result.mip_gap)
        except TimeoutError:
            record.update(status="timeout-no-incumbent")
        record["wall_time_s"] = time.monotonic() - master_started
        trace.append(record)
    if lower > best_cost + 1e-7 * max(1.0, best_cost):
        raise RuntimeError("global lower bound exceeds incumbent")
    safe_ids = safe_event_ids(model, lp, best_cost, best_ids)
    elapsed = time.monotonic() - started
    return CertifiedPoolResult(best_cost, best_ids, lower, max(0.0, (best_cost - lower) / best_cost) if best_cost else 0.0, certified, baseline, model.reachable, len(model.edges), len(safe_ids), len(last_pool), sum(cut.deadline is None for cut in lp.cuts), sum(cut.deadline is not None for cut in lp.cuts), lp.lp_objective, lp_status, lp.wall_time_s, elapsed, elapsed > time_limit, trace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--cut-mode", choices=["none", "root", "deadline"], default="deadline")
    parser.add_argument("--lp-rounds", type=int, default=4)
    parser.add_argument("--probes", type=int, default=24)
    args = parser.parse_args()
    result = certified_pool_optimize(TemporalGraph.from_json(args.input), args.root, time_limit=args.time_limit, cut_mode=args.cut_mode, max_lp_rounds=args.lp_rounds, max_probes=args.probes)
    print(json.dumps(asdict(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
