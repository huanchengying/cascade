#!/usr/bin/env python3
"""Priced warm masters initialized by exact shared-activation star descent."""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from temporal_mst_certificate import PrefixLPResult, build_prefix_model, safe_event_ids, solve_prefix_lp, support_closed_pool
from temporal_mst_cascade_star import cascade_activation_star
from temporal_mst_highs import ikuta_warm_start
from temporal_mst_prototype import TemporalGraph, validate_temporal_tree
from temporal_mst_star import shared_activation_star


def _uncoupled_weight_lp(model) -> PrefixLPResult:
    fallback = math.fsum(
        min(model.edges[position].weight for position in model.incoming[vertex])
        for vertex in model.reachable
        if vertex != model.root
    )


    excess = np.zeros(len(model.edges))
    for vertex, positions in model.incoming.items():
        if vertex == model.root:
            continue
        minimum = min(model.edges[position].weight for position in positions)
        for position in positions:
            excess[position] = model.edges[position].weight - minimum
    return PrefixLPResult(
        fallback,
        fallback,
        excess,
        np.zeros(len(model.edges)),
        [],
        0,
        "uncoupled-incoming-weight",
        0.0,
        0.0,
        0.0,
    )


def _shared_initial(
    graph: TemporalGraph,
    root: str,
    *,
    initial_mode: str,
    star_rounds: int,
    cascade_rounds: int,
    cascade_max_hubs: int,
    cascade_max_parent_evaluations: int,
):
    if initial_mode == "star":
        initial = shared_activation_star(graph, root, max_rounds=star_rounds)
        metrics = {
            "initial_algorithm": initial.algorithm,
            "star_moves": initial.accepted_moves,
            "star_children": initial.activated_children,
            "cascade_moves": 0,
            "cascade_children": 0,
            "relaxed_deadline_vertices": 0,
        }
    elif initial_mode == "cascade":
        initial = cascade_activation_star(
            graph,
            root,
            conservative_rounds=star_rounds,
            cascade_rounds=cascade_rounds,
            max_hubs=cascade_max_hubs,
            max_parent_evaluations=cascade_max_parent_evaluations,
        )
        metrics = {
            "initial_algorithm": initial.algorithm,
            "star_moves": None,
            "star_children": None,
            "cascade_moves": initial.accepted_moves,
            "cascade_children": initial.activated_children,
            "relaxed_deadline_vertices": initial.relaxed_deadline_vertices,
        }
    else:
        raise ValueError("initial_mode must be star or cascade")
    return initial, metrics


def star_warm_full(
    graph: TemporalGraph,
    root: str,
    *,
    time_limit: float = 30.0,
    star_rounds: int = 4,
    initial_mode: str = "star",
    cascade_rounds: int = 2,
    cascade_max_hubs: int = 64,
    cascade_max_parent_evaluations: int = 128,
    cut_mode: str = "none",
    max_probes: int = 6,
    max_lp_rounds: int = 3,
) -> dict:
    """Run the full native Ikuta master from a shared-activation incumbent."""

    if (
        time_limit <= 0
        or star_rounds < 0
        or cascade_rounds < 0
        or cascade_max_hubs < 1
        or cascade_max_parent_evaluations < 1
        or max_probes < 1
        or max_lp_rounds < 1
    ):
        raise ValueError("invalid shared-initial settings")
    if cut_mode not in {"none", "root"}:
        raise ValueError("full warm master supports none or root cuts")
    started = time.monotonic()
    initial, initial_metrics = _shared_initial(
        graph,
        root,
        initial_mode=initial_mode,
        star_rounds=star_rounds,
        cascade_rounds=cascade_rounds,
        cascade_max_hubs=cascade_max_hubs,
        cascade_max_parent_evaluations=cascade_max_parent_evaluations,
    )
    initial_time = time.monotonic() - started
    lp = None
    cuts = []
    lp_status = "not-run"
    lp_time = 0.0
    lower = 0.0
    if cut_mode == "root":
        model = build_prefix_model(graph, root)
        remaining = time_limit - (time.monotonic() - started)
        if remaining > 0:
            lp_started = time.monotonic()
            try:
                lp = solve_prefix_lp(
                    model,
                    cut_mode="root",
                    max_rounds=max_lp_rounds,
                    max_probes=max_probes,
                    time_limit=min(0.45 * time_limit, remaining),
                )
                cuts = lp.cuts
                lower = lp.lower_bound
                lp_status = lp.status
            except TimeoutError as exc:
                lp_status = "timeout-fallback: " + str(exc)
            lp_time = time.monotonic() - lp_started
    remaining = time_limit - (time.monotonic() - started)
    if remaining <= 0:
        relative_gap = max(0.0, (initial.cost - lower) / initial.cost) if initial.cost else 0.0
        certified = relative_gap <= 1e-7
        return {
            "cost": initial.cost,
            "edge_ids": initial.edge_ids,
            "lower_bound": lower,
            "relative_gap": relative_gap,
            "mip_gap": relative_gap,
            "optimal": certified,
            "certified_optimal": certified,
            "status": "optimal" if certified else "budget-before-solve",
            "baseline_cost": initial.baseline_cost,
            "initial_cost": initial.cost,
            "initial_time_s": initial_time,
            **initial_metrics,
            "cut_mode": cut_mode,
            "root_cuts": len(cuts),
            "lp_objective": lp.lp_objective if lp is not None else None,
            "lp_status": lp_status,
            "lp_time_s": lp_time,
            "total_time_s": time.monotonic() - started,
            "deadline_exceeded": time.monotonic() - started > time_limit,
            "solver": "highspy-1.7.2",
            "warm_start_supplied": True,
        }
    result = ikuta_warm_start(
        graph,
        root,
        time_limit=remaining,
        initial_ids=initial.edge_ids,
        cuts=cuts,
    )
    elapsed = time.monotonic() - started
    lower = max(lower, result["lower_bound"])
    relative_gap = max(0.0, (result["cost"] - lower) / result["cost"]) if result["cost"] else 0.0
    certified = relative_gap <= 1e-7
    result.update(
        lower_bound=lower,
        mip_gap=relative_gap,
        relative_gap=relative_gap,
        optimal=result["optimal"] or certified,
        certified_optimal=result["optimal"] or certified,
        status="optimal" if result["optimal"] or certified else result["status"],
        baseline_cost=initial.baseline_cost,
        initial_cost=initial.cost,
        initial_time_s=initial_time,
        **initial_metrics,
        cut_mode=cut_mode,
        root_cuts=len(cuts),
        lp_objective=lp.lp_objective if lp is not None else None,
        lp_status=lp_status,
        lp_time_s=lp_time,
        total_time_s=elapsed,
        deadline_exceeded=elapsed > time_limit,
    )
    return result


def star_priced_warm_optimize(
    graph: TemporalGraph,
    root: str,
    *,
    time_limit: float = 30.0,
    cut_mode: str = "root",
    max_probes: int = 6,
    max_lp_rounds: int = 3,
    widths: Sequence[int] = (1, 2, 4),
    safe_master_limit: int = 200_000,
    star_rounds: int = 4,
    initial_mode: str = "star",
    cascade_rounds: int = 2,
    cascade_max_hubs: int = 64,
    cascade_max_parent_evaluations: int = 128,
) -> dict:
    """Use one shared-activation incumbent in every support-closed master."""

    if (
        time_limit <= 0
        or not widths
        or min(widths) < 1
        or safe_master_limit < 0
        or star_rounds < 0
        or cascade_rounds < 0
        or cascade_max_hubs < 1
        or cascade_max_parent_evaluations < 1
    ):
        raise ValueError("invalid star-priced settings")
    if cut_mode not in {"weight", "none", "root", "deadline"}:
        raise ValueError("invalid cut mode")
    if any(edge.arrival <= edge.start for edge in graph.edges):
        raise ValueError("star-priced master requires positive durations")

    started = time.monotonic()
    initial, initial_metrics = _shared_initial(
        graph,
        root,
        initial_mode=initial_mode,
        star_rounds=star_rounds,
        cascade_rounds=cascade_rounds,
        cascade_max_hubs=cascade_max_hubs,
        cascade_max_parent_evaluations=cascade_max_parent_evaluations,
    )
    initial_time = time.monotonic() - started
    best_cost = initial.cost
    best_ids = list(initial.edge_ids)
    model = build_prefix_model(graph, root)
    fallback = math.fsum(
        min(model.edges[position].weight for position in model.incoming[vertex])
        for vertex in model.reachable
        if vertex != root
    )
    lp_started = time.monotonic()
    if cut_mode == "weight":
        lp = _uncoupled_weight_lp(model)
    else:
        try:
            remaining = time_limit - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("budget exhausted during shared-star initialization")
            lp = solve_prefix_lp(
                model,
                cut_mode=cut_mode,
                max_rounds=max_lp_rounds,
                max_probes=max_probes,
                time_limit=min(0.45 * time_limit, remaining),
            )
        except TimeoutError as exc:
            lp = PrefixLPResult(
                fallback,
                fallback,
                np.zeros(len(model.edges)),
                np.zeros(len(model.edges)),
                [],
                0,
                "timeout-fallback: " + str(exc),
                0.0,
                0.0,
                time.monotonic() - lp_started,
            )

    lower = lp.lower_bound
    safe_ids = safe_event_ids(model, lp, best_cost, best_ids)
    last_pool = set(best_ids)
    certified = best_cost - lower <= 1e-7 * max(1.0, best_cost)
    trace = []
    settings = list(widths) + [None]
    for stage, width in enumerate(settings):
        if certified or time.monotonic() - started >= time_limit:
            break
        safe_ids = safe_event_ids(model, lp, best_cost, best_ids)
        if width is None:
            if len(safe_ids) > safe_master_limit:
                continue
            pool = set(safe_ids)
        else:
            pool = support_closed_pool(model, lp, safe_ids, best_ids, width)
        last_pool = pool
        complete = pool == safe_ids
        master_graph = TemporalGraph(
            list(graph.vertices),
            [edge for edge in model.edges if edge.eid in pool],
        )
        remaining = time_limit - (time.monotonic() - started)
        if remaining <= 0:
            break
        allowance = (
            remaining
            if complete or width is None
            else remaining / (len(settings) - stage)
        )
        master_started = time.monotonic()
        result = ikuta_warm_start(
            master_graph,
            root,
            time_limit=allowance,
            initial_ids=best_ids,
            cuts=lp.cuts,
        )
        selected_ids = set(result["edge_ids"])
        selected = [edge for edge in model.edges if edge.eid in selected_ids]
        if not validate_temporal_tree(graph, root, selected, set(model.reachable)):
            raise RuntimeError("star-priced master lost a required original vertex")
        if result["cost"] < best_cost:
            best_cost = result["cost"]
            best_ids = sorted(result["edge_ids"])
        if complete:
            lower = max(lower, result["lower_bound"])
        certified = best_cost - lower <= 1e-7 * max(1.0, best_cost)
        trace.append(
            {
                "width": width,
                "pool_edges": len(pool),
                "safe_edges": len(safe_ids),
                "globally_equivalent": complete,
                "solver_allowance_s": allowance,
                "status": result["status"],
                "cost": result["cost"],
                "restricted_lower_bound": result["lower_bound"],
                "mip_gap": result["mip_gap"],
                "wall_time_s": time.monotonic() - master_started,
                "cuts_transferred": len(lp.cuts),
            }
        )

    if lower > best_cost + 1e-7 * max(1.0, best_cost):
        raise RuntimeError("star-priced global bound exceeds the incumbent")
    safe_ids = safe_event_ids(model, lp, best_cost, best_ids)
    elapsed = time.monotonic() - started
    return {
        "cost": best_cost,
        "edge_ids": best_ids,
        "lower_bound": lower,
        "relative_gap": max(0.0, (best_cost - lower) / best_cost) if best_cost else 0.0,
        "certified_optimal": certified,
        "baseline_cost": initial.baseline_cost,
        "initial_cost": initial.cost,
        "initial_time_s": initial_time,
        **initial_metrics,
        "reachable": model.reachable,
        "original_edges": len(model.edges),
        "safe_edges": len(safe_ids),
        "pool_edges": len(last_pool),
        "root_cuts": sum(cut.deadline is None for cut in lp.cuts),
        "deadline_cuts": sum(cut.deadline is not None for cut in lp.cuts),
        "lp_objective": lp.lp_objective,
        "lp_status": lp.status,
        "lp_time_s": lp.wall_time_s,
        "total_time_s": elapsed,
        "deadline_exceeded": elapsed > time_limit,
        "master_trace": trace,
        "algorithm": "shared-star-priced-support-closure",
        "solver": "highspy-1.7.2",
        "warm_start_supplied": True,
    }
