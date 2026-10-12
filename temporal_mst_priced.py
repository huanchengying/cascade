#!/usr/bin/env python3
"""Support-closed priced masters on the same warm-start engine as the baseline.

Deadline cuts are a candidate strengthening; min-cut separation and reduced-cost
fixing are standard. A restricted-master dual bound is never a global bound.
"""

import math
import time
from dataclasses import asdict
from typing import Sequence

import numpy as np

from temporal_mst_certificate import (
    CertifiedPoolResult, PrefixLPResult, build_prefix_model, safe_event_ids,
    solve_prefix_lp, support_closed_pool,
)
from temporal_mst_highs import ikuta_warm_start
from temporal_mst_prototype import TemporalGraph, validate_temporal_tree
from temporal_mst_shared import shortest_path_union_tree


def priced_warm_optimize(
    graph: TemporalGraph,
    root: str,
    *,
    time_limit: float = 30.0,
    cut_mode: str = "deadline",
    max_probes: int = 6,
    max_lp_rounds: int = 3,
    widths: Sequence[int] = (1, 2, 4),
    safe_master_limit: int = 200_000,
) -> dict:
    if time_limit <= 0 or not widths or min(widths) < 1 or safe_master_limit < 0:
        raise ValueError("invalid priced-master settings")
    if any(edge.arrival <= edge.start for edge in graph.edges):
        raise ValueError("priced master requires positive durations")
    started = time.monotonic()
    model = build_prefix_model(graph, root)
    baseline, tree, _ = shortest_path_union_tree(graph, root)
    best_cost = baseline
    best_ids = sorted(edge.eid for edge in tree)
    fallback = math.fsum(min(model.edges[position].weight for position in model.incoming[vertex]) for vertex in model.reachable if vertex != root)
    lp_started = time.monotonic()
    if cut_mode == "weight":
        excess = np.zeros(len(model.edges))
        for vertex, positions in model.incoming.items():
            if vertex == root:
                continue
            minimum = min(model.edges[position].weight for position in positions)
            for position in positions:
                excess[position] = model.edges[position].weight - minimum
        lp = PrefixLPResult(fallback, fallback, excess, np.zeros(len(model.edges)), [], 0, "uncoupled-incoming-weight", 0.0, 0.0, 0.0)
    else:
        try:
            remaining = time_limit - (lp_started - started)
            if remaining <= 0:
                raise TimeoutError("budget exhausted during preparation")
            lp = solve_prefix_lp(model, cut_mode=cut_mode, max_rounds=max_lp_rounds, max_probes=max_probes, time_limit=min(0.45 * time_limit, remaining))
        except TimeoutError as exc:
            lp = PrefixLPResult(fallback, fallback, np.zeros(len(model.edges)), np.zeros(len(model.edges)), [], 0, "timeout-fallback: " + str(exc), 0.0, 0.0, time.monotonic() - lp_started)
    lower = lp.lower_bound
    safe_ids = safe_event_ids(model, lp, best_cost, best_ids)
    last_pool = set(best_ids)
    certified = best_cost - lower <= 1e-7 * max(1.0, best_cost)
    trace = []
    settings = list(widths) + [None]
    for stage, width in enumerate(settings):
        if certified or time.monotonic() - started >= time_limit:
            break
        master_started = time.monotonic()
        safe_ids = safe_event_ids(model, lp, best_cost, best_ids)
        if width is None:
            if len(safe_ids) > safe_master_limit:
                continue
            pool = set(safe_ids)
        else:
            pool = support_closed_pool(model, lp, safe_ids, best_ids, width)
        last_pool = pool
        master_graph = TemporalGraph(list(graph.vertices), [edge for edge in model.edges if edge.eid in pool])
        remaining = time_limit - (time.monotonic() - started)
        if remaining <= 0:
            break
        complete = pool == safe_ids
        allowance = remaining if complete or width is None else remaining / (len(settings) - stage)
        result = ikuta_warm_start(master_graph, root, time_limit=allowance, initial_ids=best_ids, cuts=lp.cuts)
        selected_ids = set(result["edge_ids"])
        selected = [edge for edge in model.edges if edge.eid in selected_ids]
        if not validate_temporal_tree(graph, root, selected, set(model.reachable)):
            raise RuntimeError("priced master lost a required original-graph vertex")
        if result["cost"] < best_cost:
            best_cost, best_ids = result["cost"], sorted(result["edge_ids"])
        if complete:
            lower = max(lower, result["lower_bound"])
        certified = best_cost - lower <= 1e-7 * max(1.0, best_cost)
        trace.append({"width": width, "pool_edges": len(pool), "safe_edges": len(safe_ids), "globally_equivalent": complete, "solver_allowance_s": allowance, "status": result["status"], "cost": result["cost"], "restricted_lower_bound": result["lower_bound"], "mip_gap": result["mip_gap"], "wall_time_s": time.monotonic() - master_started, "cuts_transferred": len(lp.cuts)})
    if lower > best_cost + 1e-7 * max(1.0, best_cost):
        raise RuntimeError("priced global bound exceeds the incumbent")
    safe_ids = safe_event_ids(model, lp, best_cost, best_ids)
    elapsed = time.monotonic() - started
    result = CertifiedPoolResult(best_cost, best_ids, lower, max(0.0, (best_cost - lower) / best_cost) if best_cost else 0.0, certified, baseline, model.reachable, len(model.edges), len(safe_ids), len(last_pool), sum(cut.deadline is None for cut in lp.cuts), sum(cut.deadline is not None for cut in lp.cuts), lp.lp_objective, lp.status, lp.wall_time_s, elapsed, elapsed > time_limit, trace, "warm-priced-support-closure")
    return dict(asdict(result), solver="highspy-1.7.2", warm_start_supplied=True)
