#!/usr/bin/env python3
"""Native HiGHS Ikuta master with a validated initial physical temporal tree.

Positive-duration model only. Initial tree support replicates an important
aspect of Ikuta's primal initialization missing from scipy.optimize.milp.
This does not implement the authors' CPLEX node-level fractional callbacks.
"""

import math
import time
from typing import Optional, Sequence

import highspy
import numpy as np
from scipy.sparse import vstack

from temporal_mst_certificate import ConnectivityCut, _cut_matrix, build_prefix_model
from temporal_mst_prototype import TemporalGraph, earliest_arrival_tree, validate_temporal_tree


def ikuta_warm_start(
    graph: TemporalGraph,
    root: str,
    *,
    time_limit: float = 30.0,
    initial_ids: Optional[Sequence[str]] = None,
    cuts: Sequence[ConnectivityCut] = (),
) -> dict:
    if time_limit <= 0 or any(edge.arrival <= edge.start for edge in graph.edges):
        raise ValueError("native warm-start master requires positive duration and budget")
    started = time.monotonic()
    model = build_prefix_model(graph, root)
    if initial_ids is None:
        _, parent = earliest_arrival_tree(graph, root)
        initial_ids = [edge.eid for edge in parent.values()]
    ids = set(initial_ids)
    if not ids <= {edge.eid for edge in model.edges}:
        raise ValueError("initial_ids must encode a valid spanning temporal tree")
    initial = [edge for edge in model.edges if edge.eid in ids]
    if not validate_temporal_tree(graph, root, initial, set(model.reachable)):
        raise ValueError("initial_ids must encode a valid spanning temporal tree")
    initial_cost = math.fsum(edge.weight for edge in initial)
    edge_count = len(model.edges)
    if edge_count == 0:
        return {"cost": 0.0, "edge_ids": [], "lower_bound": 0.0, "mip_gap": 0.0, "optimal": True, "status": "optimal", "initial_cost": 0.0, "solver": "highspy-1.7.2"}
    support, support_rhs = _cut_matrix(model, cuts)
    matrix = vstack((model.equality, support), format="csr")
    row_lower = np.concatenate((model.rhs, np.full(len(support_rhs), -math.inf)))
    row_upper = np.concatenate((model.rhs, support_rhs))
    highs_model = highspy.HighsModel()
    linear = highs_model.lp_
    linear.num_col_ = len(model.objective)
    linear.num_row_ = len(row_lower)
    linear.col_cost_ = model.objective
    linear.col_lower_ = np.zeros(len(model.objective))
    linear.col_upper_ = np.ones(len(model.objective))
    linear.row_lower_ = row_lower
    linear.row_upper_ = row_upper
    linear.integrality_ = [highspy.HighsVarType.kInteger] * len(model.objective)
    linear.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    linear.a_matrix_.start_ = matrix.indptr.astype(np.int32)
    linear.a_matrix_.index_ = matrix.indices.astype(np.int32)
    linear.a_matrix_.value_ = matrix.data
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("threads", 1)
    solver.setOptionValue("random_seed", 0)
    solver.setOptionValue("mip_rel_gap", 0.0)
    solver.passModel(highs_model)
    values = np.zeros(len(model.objective))
    values[:edge_count] = [float(edge.eid in ids) for edge in model.edges]
    for positions in model.incoming.values():
        mass = 0.0
        for position in positions:
            mass += values[position]
            values[edge_count + position] = mass
    solution = highspy.HighsSolution()
    solution.col_value = values
    solution.row_value = matrix @ values
    solution.value_valid = True
    if solver.setSolution(solution) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS rejected a validated MIP start")
    remaining = time_limit - (time.monotonic() - started)
    if remaining <= 0:
        return {"cost": initial_cost, "edge_ids": sorted(ids), "lower_bound": 0.0, "mip_gap": 1.0, "optimal": False, "status": "budget-before-solve", "initial_cost": initial_cost, "solver": "highspy-1.7.2"}
    solver.setOptionValue("time_limit", remaining)
    solver.run()
    info = solver.getInfo()
    final = solver.getSolution()
    final_values = final.col_value
    chosen = [edge for position, edge in enumerate(model.edges) if final_values[position] >= 0.5] if final.value_valid else initial
    if not validate_temporal_tree(graph, root, chosen, set(model.reachable)):
        raise RuntimeError("native HiGHS incumbent violates temporal-tree constraints")
    cost = math.fsum(edge.weight for edge in chosen)
    lower = max(0.0, info.mip_dual_bound) if math.isfinite(info.mip_dual_bound) else 0.0
    optimal = solver.getModelStatus() == highspy.HighsModelStatus.kOptimal and info.mip_gap <= 1e-9
    return {"cost": cost, "edge_ids": sorted(edge.eid for edge in chosen), "lower_bound": lower, "mip_gap": max(0.0, (cost - lower) / cost) if cost else 0.0, "optimal": optimal, "status": "optimal" if optimal else "feasible", "initial_cost": initial_cost, "warm_start_supplied": True, "solve_time_s": time.monotonic() - started, "solver": "highspy-" + solver.version()}
