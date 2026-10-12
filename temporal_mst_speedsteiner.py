#!/usr/bin/env python3
"""Temporal adapter for the frozen SpeedSteiner authors' C++ implementation.

The upstream solver uses float weights. Cost is recomputed in original double
precision. Parallel equal-state arcs are safely collapsed to the cheapest one.
"""

import math
import subprocess
import time
from pathlib import Path

from temporal_mst_prototype import TimeStateDAG, _tree_from_edge_ids, validate_temporal_tree


def speedsteiner_tree(graph, root, *, time_limit=30.0, method="fast_level2", binary=None, max_state_terminal_product=50_000_000):
    if time_limit <= 0 or method not in {"level2", "fast_level2"}:
        raise ValueError("invalid SpeedSteiner settings")
    if any(edge.arrival <= edge.start for edge in graph.edges):
        raise ValueError("SpeedSteiner adapter requires positive durations")
    started = time.monotonic()
    dag = TimeStateDAG(graph, root)
    nodes = sorted(dag.adj, key=lambda node: (node != dag.source, node))
    terminals = sorted(dag.terminals.values())
    if len(nodes) * len(terminals) > max_state_terminal_product:
        raise ValueError(f"SpeedSteiner memory guard: {len(nodes)} states x {len(terminals)} terminals")
    if not terminals:
        return {"cost": 0.0, "edge_ids": [], "upstream_float_cost": 0.0, "state_count": len(nodes), "terminals": 0, "wall_time_s": time.monotonic() - started}
    node_ids = {node: position for position, node in enumerate(nodes)}
    arcs = {}
    for tail, outgoing in dag.adj.items():
        for head, cost, physical in outgoing:
            key = node_ids[tail], node_ids[head]
            candidate = cost, physical or ""
            if key not in arcs or candidate < (arcs[key][0], arcs[key][1] or ""):
                arcs[key] = cost, physical
    lines = [f"{node_ids[dag.source]} {len(arcs)} {len(terminals)}", " ".join(str(node_ids[terminal]) for terminal in terminals)]
    lines.extend(f"{tail} {head} {cost:.17g}" for (tail, head), (cost, _) in sorted(arcs.items()))
    remaining = time_limit - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("SpeedSteiner budget exhausted by temporal transformation")
    executable = Path(binary) if binary else Path(__file__).with_name("work") / "bin" / "speedsteiner_adapter"
    if not executable.is_file():
        raise RuntimeError("build SpeedSteiner adapter before using this baseline")
    try:
        process = subprocess.run([str(executable.resolve()), method], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("SpeedSteiner author binary exceeded wall-time budget") from exc
    if process.returncode:
        raise RuntimeError(f"SpeedSteiner failed ({process.returncode}): {process.stderr[-2000:]}")
    output = process.stdout.splitlines()
    if len(output) < 2 or len(output) - 2 != int(output[1]):
        raise RuntimeError("invalid SpeedSteiner adapter output")
    selected = set()
    for line in output[2:]:
        tail, head = map(int, line.split())
        if (tail, head) not in arcs:
            raise RuntimeError("upstream solution uses an unknown state arc")
        physical = arcs[tail, head][1]
        if physical is not None:
            selected.add(physical)
    tree = _tree_from_edge_ids(graph, root, sorted(selected), dag.reachable)
    if not validate_temporal_tree(graph, root, tree, dag.reachable):
        raise RuntimeError("SpeedSteiner expansion is not a temporal spanning tree")
    return {"cost": math.fsum(edge.weight for edge in tree), "edge_ids": sorted(edge.eid for edge in tree), "upstream_float_cost": float(output[0]), "state_count": len(nodes), "terminals": len(terminals), "state_arcs": len(arcs), "wall_time_s": time.monotonic() - started, "algorithm": "speedsteiner-author-" + method, "upstream_commit": "4191b7cf54b9310f065540dd80fc587aeefd3033"}
