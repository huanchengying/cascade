#!/usr/bin/env python3
"""Small, dependency-free temporal MST research prototype.

The prototype deliberately exposes three different objects:
  1. earliest_arrival_tree: Huang-style MST_a baseline;
  2. TimeStateDAG/TeGraphDAG + path-union: explicit DAG upper bounds;
  3. exact_temporal_tree: small-instance branch-and-bound verifier for MST_w.

The DAG solver is a research scaffold, not a claim of a polynomial algorithm for
general MST_w.  General MST_w remains NP-hard; the code is intended for proofs,
counterexamples, and exact cross-checks on small instances.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class TemporalEdge:
    eid: str
    u: str
    v: str
    start: float
    arrival: float
    weight: float


@dataclass
class TemporalGraph:
    vertices: List[str]
    edges: List[TemporalEdge]

    def __post_init__(self) -> None:
        if len(self.vertices) != len(set(self.vertices)):
            raise ValueError("vertices must be unique")
        vertex_set = set(self.vertices)
        edge_ids: Set[str] = set()
        for e in self.edges:
            if e.eid in edge_ids:
                raise ValueError(f"duplicate temporal edge id: {e.eid}")
            edge_ids.add(e.eid)
            if e.u not in vertex_set or e.v not in vertex_set:
                raise ValueError(f"edge endpoint not in vertices: {e.eid}")
            if e.arrival < e.start:
                raise ValueError(f"arrival < start for temporal edge: {e.eid}")
            if not all(math.isfinite(x) for x in (e.start, e.arrival, e.weight)):
                raise ValueError(f"non-finite temporal edge value: {e.eid}")
            if e.weight < 0:
                raise ValueError(f"negative edge weight is unsupported: {e.eid}")

    @staticmethod
    def from_json(path: str) -> "TemporalGraph":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        edges = []
        seen_ids: Set[str] = set()
        for i, item in enumerate(raw["edges"]):
            eid = str(item.get("id", f"e{i}"))
            if eid in seen_ids:
                raise ValueError(f"duplicate temporal edge id: {eid}")
            seen_ids.add(eid)
            start = float(item["start"])
            arrival = float(item["arrival"])
            weight = float(item["weight"])
            if not (math.isfinite(start) and math.isfinite(arrival) and math.isfinite(weight)):
                raise ValueError(f"non-finite temporal edge value: {eid}")
            if arrival < start:
                raise ValueError(f"arrival < start for temporal edge: {eid}")
            edges.append(
                TemporalEdge(
                    eid,
                    str(item["u"]),
                    str(item["v"]),
                    start,
                    arrival,
                    weight,
                )
            )
        vertices = [str(v) for v in raw["vertices"]]
        if len(vertices) != len(set(vertices)):
            raise ValueError("vertices must be unique")
        vertex_set = set(vertices)
        for e in edges:
            if e.u not in vertex_set or e.v not in vertex_set:
                raise ValueError(f"edge endpoint not in vertices: {e.eid}")
        return TemporalGraph(vertices, edges)

    def outgoing(self, u: str) -> List[TemporalEdge]:
        return [e for e in self.edges if e.u == u]


def _earliest_arrival_from_ordered_edges(
    vertices: Sequence[str],
    ordered: Sequence[TemporalEdge],
    root: str,
) -> Tuple[Dict[str, float], Dict[str, TemporalEdge]]:
    if root not in vertices:
        raise ValueError(f"unknown root vertex: {root}")
    inf = math.inf
    arrival = {v: inf for v in vertices}
    parent: Dict[str, TemporalEdge] = {}
    arrival[root] = -math.inf
    changed = True
    # A fixed point is useful when a zero-duration/equal-time edge chain exists.
    # With strictly increasing arrivals, one pass is sufficient after sorting.
    while changed:
        changed = False
        for e in ordered:
            if arrival.get(e.u, inf) <= e.start and e.arrival < arrival.get(e.v, inf):
                arrival[e.v] = e.arrival
                parent[e.v] = e
                changed = True
    return arrival, parent


def earliest_arrival_tree(graph: TemporalGraph, root: str) -> Tuple[Dict[str, float], Dict[str, TemporalEdge]]:
    """Return earliest arrival labels and parent edges.

    Edges are scanned in nondecreasing start/arrival order.  This is the
    time-respecting single-source baseline, not the weight-minimum MST_w.
    """
    ordered = sorted(graph.edges, key=lambda e: (e.start, e.arrival, e.eid))
    return _earliest_arrival_from_ordered_edges(graph.vertices, ordered, root)


@dataclass(frozen=True)
class State:
    sid: str
    vertex: str
    time: float
    edge_id: Optional[str]


class TimeStateDAG:
    """Huang-style time-state DAG with linear-size explicit materialization.

    For every vertex v, one state is created for each *distinct* arrival time
    of an incoming temporal edge.  States are linked in increasing time order
    by zero-cost waiting arcs, and the last state links to terminal T:v.  Each
    temporal edge contributes one real arc from the latest reachable state of
    its tail (or the root source) to the state of its head.  Equal-time states
    are merged, which preserves non-strict (``arrival <= start``) temporal
    paths without introducing zero-time cycles.

    This is the O(|E| + |V|) reduction used as the baseline in Huang et al.; a
    production TeGraph implementation can keep the same semantics while
    representing the waiting chains as compressed hyper-edges.
    """

    def __init__(self, graph: TemporalGraph, root: str):
        self.graph = graph
        self.root = root
        if root not in graph.vertices:
            raise ValueError(f"unknown root vertex: {root}")
        self.states: Dict[str, State] = {}
        self.edge_state: Dict[str, str] = {}
        base_arrival, _ = earliest_arrival_tree(graph, root)
        self.reachable = {v for v, t in base_arrival.items() if t < math.inf}
        self.terminals = {
            v: f"T:{v}" for v in graph.vertices if v != root and v in self.reachable
        }
        self.adj: Dict[str, List[Tuple[str, float, Optional[str]]]] = {}
        self.source = f"R:{root}"
        self._topological_order: List[str] = []
        self._build()

    def _build(self) -> None:
        incoming: Dict[str, List[TemporalEdge]] = {v: [] for v in self.graph.vertices}
        for e in self.graph.edges:
            # An edge entering the root cannot occur in a rooted tree: it would
            # create an undirected cycle or give the root a second parent.
            if e.v != self.root:
                incoming[e.v].append(e)

        self.adj[self.source] = []
        state_times: Dict[str, List[float]] = {}
        state_index: Dict[str, Dict[float, int]] = {}
        for v, edges in incoming.items():
            times = sorted({e.arrival for e in edges})
            state_times[v] = times
            state_index[v] = {time: i for i, time in enumerate(times)}
            for i, time in enumerate(times):
                sid = f"Q:{v}:{i}"
                self.states[sid] = State(sid, v, time, None)
                self.adj[sid] = []
            for e in edges:
                i = state_index[v][e.arrival]
                self.edge_state[e.eid] = f"Q:{v}:{i}"

            for i in range(len(times) - 1):
                self.adj[f"Q:{v}:{i}"].append((f"Q:{v}:{i + 1}", 0.0, None))
            if times and v in self.terminals:
                self.adj[f"Q:{v}:{len(times) - 1}"].append((self.terminals[v], 0.0, None))

        for v, terminal in self.terminals.items():
            self.adj.setdefault(terminal, [])

        for e in self.graph.edges:
            if e.v == self.root:
                continue
            head_state = self.edge_state[e.eid]
            if e.u == self.root:
                tail_state = self.source
            else:
                times = state_times[e.u]
                i = bisect_right(times, e.start) - 1
                if i < 0:
                    continue
                tail_state = f"Q:{e.u}:{i}"
            self.adj[tail_state].append((head_state, e.weight, e.eid))

        self._topological_order = self._compute_topological_order()

    def _compute_topological_order(self) -> List[str]:
        """Topologically sort the explicit transformation and reject zero-time cycles."""
        nodes = set(self.adj)
        for arcs in self.adj.values():
            nodes.update(v for v, _, _ in arcs)
        indegree = {u: 0 for u in nodes}
        for arcs in self.adj.values():
            for v, _, _ in arcs:
                indegree[v] += 1
        queue = deque(u for u, degree in indegree.items() if degree == 0)
        order: List[str] = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _, _ in self.adj.get(u, []):
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)
        if len(order) != len(nodes):
            raise ValueError(
                "time-state transformation contains a cycle; "
                "use strictly positive durations or a zero-time SCC reduction"
            )
        return order

    def summary(self) -> Dict[str, int]:
        return {
            "states": len(self.states),
            "terminals": len(self.terminals),
            "arcs": sum(len(xs) for xs in self.adj.values()),
        }

    def topological_order(self) -> List[str]:
        return list(self._topological_order)


class TeGraphDAG:
    """TeGraph's in/out-vertex transformation.

    Every distinct endpoint-time pair creates out(u,s) and in(v,a), and each
    temporal edge joins its corresponding states by a real arc of cost w.  An
    in-state connects to the first usable out-state
    of the same vertex with a zero-cost waiting arc.  Waiting uses
    ``start >= arrival`` to match Huang's non-strict temporal-path convention.
    With zero-duration cycles, no time-expanded representation is a DAG
    without an additional SCC/tie-order reduction; such input is rejected by
    the cycle check below.
    """

    def __init__(self, graph: TemporalGraph, root: str):
        if root not in graph.vertices:
            raise ValueError(f"unknown root vertex: {root}")
        self.graph = graph
        self.root = root
        base_arrival, _ = earliest_arrival_tree(graph, root)
        self.reachable = {v for v, t in base_arrival.items() if t < math.inf}
        self.terminals = {
            v: f"T:{v}" for v in graph.vertices if v != root and v in self.reachable
        }
        self.source = f"R:{root}"
        self.states: Dict[str, State] = {}
        self.adj: Dict[str, List[Tuple[str, float, Optional[str]]]] = {
            self.source: []
        }
        self._build()
        self._topological_order = self._compute_topological_order()

    @staticmethod
    def _out_id(vertex: str, index: int) -> str:
        return f"O:{vertex}:{index}"

    @staticmethod
    def _in_id(vertex: str, index: int) -> str:
        return f"I:{vertex}:{index}"

    def _build(self) -> None:
        out_times: Dict[str, List[float]] = {v: [] for v in self.graph.vertices}
        in_times: Dict[str, List[float]] = {v: [] for v in self.graph.vertices}
        for e in self.graph.edges:
            if e.v == self.root:
                continue
            out_times[e.u].append(e.start)
            in_times[e.v].append(e.arrival)
        for v in self.graph.vertices:
            out_times[v] = sorted(set(out_times[v]))
            in_times[v] = sorted(set(in_times[v]))
            for i, t in enumerate(out_times[v]):
                sid = self._out_id(v, i)
                self.states[sid] = State(sid, v, t, None)
                self.adj[sid] = []
            for i, t in enumerate(in_times[v]):
                sid = self._in_id(v, i)
                self.states[sid] = State(sid, v, t, None)
                self.adj[sid] = []

        out_index = {v: {t: i for i, t in enumerate(times)} for v, times in out_times.items()}
        in_index = {v: {t: i for i, t in enumerate(times)} for v, times in in_times.items()}

        # A terminal is attached once per distinct in-state.  Adding it once
        # per temporal edge is semantically redundant and inflates both the
        # representation and transformation cost when events share an arrival
        # timestamp.
        terminal_linked: Set[str] = set()
        for e in self.graph.edges:
            if e.v == self.root:
                continue
            oi = out_index[e.u][e.start]
            ii = in_index[e.v][e.arrival]
            out_node = self._out_id(e.u, oi)
            in_node = self._in_id(e.v, ii)
            self.adj[out_node].append((in_node, e.weight, e.eid))
            if e.v in self.terminals and in_node not in terminal_linked:
                self.adj[in_node].append((self.terminals[e.v], 0.0, None))
                terminal_linked.add(in_node)

        for v in self.graph.vertices:
            outs = out_times[v]
            ins = in_times[v]
            # Once a path reaches an out-state, it may wait through later
            # out-states.  This chain is the sparse representation of arbitrary
            # waiting and keeps the number of virtual arcs linear.
            for i in range(len(outs) - 1):
                self.adj[self._out_id(v, i)].append(
                    (self._out_id(v, i + 1), 0.0, None)
                )
            for i, t in enumerate(ins):
                j = bisect_left(outs, t)
                if j < len(outs):
                    self.adj[self._in_id(v, i)].append(
                        (self._out_id(v, j), 0.0, None)
                    )

        for v, terminal in self.terminals.items():
            self.adj.setdefault(terminal, [])
        if out_times[self.root]:
            self.adj[self.source].append((self._out_id(self.root, 0), 0.0, None))

    def _compute_topological_order(self) -> List[str]:
        nodes = set(self.adj)
        for arcs in self.adj.values():
            nodes.update(v for v, _, _ in arcs)
        indegree = {u: 0 for u in nodes}
        for arcs in self.adj.values():
            for v, _, _ in arcs:
                indegree[v] += 1
        queue = deque(u for u, degree in indegree.items() if degree == 0)
        order: List[str] = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _, _ in self.adj.get(u, []):
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)
        if len(order) != len(nodes):
            raise ValueError(
                "TeGraph transformation contains a cycle; "
                "use positive durations or the Huang state-chain mode"
            )
        return order

    def summary(self) -> Dict[str, int]:
        return {
            "states": len(self.states),
            "terminals": len(self.terminals),
            "arcs": sum(len(xs) for xs in self.adj.values()),
        }

    def topological_order(self) -> List[str]:
        return list(self._topological_order)


class LazyTeGraphIndex:
    """A hyper-DAG index with implicit zero-cost waiting transitions.

    This is the proposed systems-side representation.  It has the same
    in/out-state semantics as :class:`TeGraphDAG`, but never materializes
    ``O(v,i) -> O(v,i+1)`` or ``I(v,i) -> O(v,j)`` edges.  Instead, each
    vertex owns sorted input/output time channels; a consumer asks for the
    first output state with ``start >= arrival`` via binary search.  Physical
    temporal edges remain individually addressable, which is essential when a
    future DST solver charges selected temporal edges exactly once.

    The class is an index, not a general MST_w solver.  Its summary reports
    both the stored representation and the number of virtual arcs avoided
    relative to the explicit TeGraph-style transformation.
    """

    def __init__(self, graph: TemporalGraph, root: str):
        if root not in graph.vertices:
            raise ValueError(f"unknown root vertex: {root}")
        self.graph = graph
        self.root = root
        self.ordered_edges = sorted(
            graph.edges, key=lambda edge: (edge.start, edge.arrival, edge.eid)
        )
        arrival, _ = _earliest_arrival_from_ordered_edges(
            graph.vertices, self.ordered_edges, root
        )
        self.reachable = {v for v, t in arrival.items() if t < math.inf}
        self.out_times: Dict[str, List[float]] = {v: [] for v in graph.vertices}
        self.in_times: Dict[str, List[float]] = {v: [] for v in graph.vertices}
        self.real_arc_count = 0
        self._build()

    def _build(self) -> None:
        for e in self.graph.edges:
            if e.v == self.root:
                continue
            self.out_times[e.u].append(e.start)
            self.in_times[e.v].append(e.arrival)
            self.real_arc_count += 1
        for v in self.graph.vertices:
            self.out_times[v] = sorted(set(self.out_times[v]))
            self.in_times[v] = sorted(set(self.in_times[v]))

    def first_usable_out_state(self, vertex: str, arrival: float) -> Optional[float]:
        """Return the earliest departure state usable after ``arrival``."""
        times = self.out_times[vertex]
        index = bisect_left(times, arrival)
        return times[index] if index < len(times) else None

    def summary(self) -> Dict[str, int]:
        out_states = sum(len(times) for times in self.out_times.values())
        in_states = sum(len(times) for times in self.in_times.values())
        output_chain_arcs = sum(
            max(0, len(times) - 1) for times in self.out_times.values()
        )
        input_wait_arcs = 0
        terminal_arcs = 0
        for v in self.graph.vertices:
            output_times = self.out_times[v]
            for time in self.in_times[v]:
                if bisect_left(output_times, time) < len(output_times):
                    input_wait_arcs += 1
                if v != self.root and v in self.reachable:
                    terminal_arcs += 1
        # TeGraph's explicit layout also adds R:root -> first O(root, .).
        source_arc = int(bool(self.out_times[self.root]))
        explicit_virtual_arcs = (
            output_chain_arcs + input_wait_arcs + terminal_arcs + source_arc
        )
        return {
            "states": out_states + in_states,
            "real_arcs": self.real_arc_count,
            "materialized_virtual_arcs": 0,
            "implicit_wait_channels": sum(bool(times) for times in self.out_times.values()),
            "implicit_terminal_channels": len(self.reachable - {self.root}),
            "tegraph_explicit_virtual_arcs_avoided": explicit_virtual_arcs,
        }


def lazy_shortest_path_tree_upper_bound(
    index: LazyTeGraphIndex,
) -> Tuple[float, List[str]]:
    """Solve one shortest root-to-terminal path per terminal on the lazy DAG.

    The event sweep is equivalent to shortest-path relaxation on
    :class:`TeGraphDAG`, but waiting transitions are represented by prefix
    minima over each vertex's sorted output channel.  In-state arrivals are
    processed before output states at the same timestamp, matching the
    non-strict temporal rule ``arrival <= start``.  The result is still a
    feasible upper bound, not an exact Directed-Steiner solution.
    """
    graph = index.graph
    out_index = {
        v: {time: i for i, time in enumerate(times)}
        for v, times in index.out_times.items()
    }
    in_index = {
        v: {time: i for i, time in enumerate(times)}
        for v, times in index.in_times.items()
    }
    out_dist = {
        v: [math.inf] * len(times) for v, times in index.out_times.items()
    }
    out_prev: Dict[str, List[Optional[Tuple[str, Any]]]] = {
        v: [None] * len(times) for v, times in index.out_times.items()
    }
    in_dist = {
        v: [math.inf] * len(times) for v, times in index.in_times.items()
    }
    in_prev: Dict[str, List[Optional[Tuple[str, Any]]]] = {
        v: [None] * len(times) for v, times in index.in_times.items()
    }
    edges_by_start: Dict[float, List[TemporalEdge]] = {}
    for edge in graph.edges:
        if edge.v == index.root:
            continue
        edges_by_start.setdefault(edge.start, []).append(edge)

    # The three-state event order is essential at equal timestamps:
    # arrivals/in->out waiting must be visible before out->in edges depart.
    # Keep only events that actually occur at a time; scanning every vertex for
    # every timestamp would destroy the point of the lazy representation.
    in_events: Dict[float, List[Tuple[str, int]]] = {}
    out_events: Dict[float, List[Tuple[str, int]]] = {}
    for vertex, times in index.in_times.items():
        for pos, time in enumerate(times):
            in_events.setdefault(time, []).append((vertex, pos))
    for vertex, times in index.out_times.items():
        for pos, time in enumerate(times):
            out_events.setdefault(time, []).append((vertex, pos))
    event_times = sorted(set(edges_by_start) | set(in_events) | set(out_events))
    if index.out_times[index.root]:
        root_time = index.out_times[index.root][0]
        out_dist[index.root][0] = 0.0
        out_prev[index.root][0] = ("source", None)

    for current_time in event_times:
        for vertex, in_pos in in_events.get(current_time, []):
            if in_dist[vertex][in_pos] == math.inf:
                continue
            out_pos = bisect_left(index.out_times[vertex], current_time)
            if out_pos >= len(index.out_times[vertex]):
                continue
            candidate = in_dist[vertex][in_pos]
            if candidate < out_dist[vertex][out_pos]:
                out_dist[vertex][out_pos] = candidate
                out_prev[vertex][out_pos] = ("in", (vertex, in_pos))

        for vertex, out_pos in out_events.get(current_time, []):
            if out_dist[vertex][out_pos] == math.inf:
                continue
            if out_pos + 1 < len(index.out_times[vertex]):
                if out_dist[vertex][out_pos] < out_dist[vertex][out_pos + 1]:
                    out_dist[vertex][out_pos + 1] = out_dist[vertex][out_pos]
                    out_prev[vertex][out_pos + 1] = (
                        "out",
                        (vertex, out_pos),
                    )
        for edge in edges_by_start.get(current_time, []):
            source_pos = out_index[edge.u][edge.start]
            if out_dist[edge.u][source_pos] == math.inf:
                continue
            target_pos = in_index[edge.v][edge.arrival]
            candidate = out_dist[edge.u][source_pos] + edge.weight
            if candidate < in_dist[edge.v][target_pos]:
                in_dist[edge.v][target_pos] = candidate
                in_prev[edge.v][target_pos] = (
                    "edge",
                    (edge.u, source_pos, edge.eid),
                )

    def terminal_path(vertex: str) -> Tuple[float, List[str]]:
        best_pos = min(
            range(len(in_dist[vertex])),
            key=lambda pos: in_dist[vertex][pos],
            default=-1,
        )
        if best_pos < 0 or in_dist[vertex][best_pos] == math.inf:
            return math.inf, []
        path: List[str] = []
        node_kind = "in"
        node: Tuple[str, int] = (vertex, best_pos)
        while True:
            if node_kind == "in":
                previous = in_prev[node[0]][node[1]]
                if previous is None:
                    return math.inf, []
                kind, payload = previous
                if kind == "edge":
                    source_vertex, source_pos, edge_id = payload
                    path.append(edge_id)
                    node_kind = "out"
                    node = (source_vertex, source_pos)
                else:
                    return math.inf, []
            else:
                previous = out_prev[node[0]][node[1]]
                if previous is None:
                    return math.inf, []
                kind, payload = previous
                if kind == "source":
                    break
                node_kind = kind
                node = payload
        path.reverse()
        return in_dist[vertex][best_pos], path

    union: List[str] = []
    for vertex in sorted(index.reachable - {index.root}):
        cost, path = terminal_path(vertex)
        if cost == math.inf:
            return math.inf, []
        for edge_id in path:
            if edge_id not in union:
                union.append(edge_id)
    tree = _tree_from_edge_ids(graph, index.root, union, index.reachable)
    if len(tree) != len(index.reachable) - 1:
        return math.inf, []
    return sum(edge.weight for edge in tree), [edge.eid for edge in tree]


def lazy_label_sweep_upper_bound(
    index: LazyTeGraphIndex,
) -> Tuple[float, List[str]]:
    """Fast lazy-DAG upper bound using a single temporal edge sweep.

    For a shortest root-to-terminal path, all waiting transitions at one
    vertex can be summarized by the cheapest label whose arrival time is no
    later than the current departure.  We therefore keep one active prefix
    minimum per vertex and an immutable label for every successful edge
    relaxation.  The immutable labels are important: a later cheaper path to
    an intermediate vertex must not rewrite the parent of a previously used
    edge and make its reconstructed temporal path invalid.

    Complexity is ``O(M log M + L log L)`` in the general case, where ``M`` is
    the number of temporal edges and ``L <= M`` is the number of generated
    labels.  For an input already ordered by start time, the first term is
    linear.  This computes the same one-shortest-path-per-terminal feasible
    upper bound as the explicit DAG constructor, but without state arrays,
    event dictionaries, or materialized waiting arcs.
    """
    graph = index.graph
    edges = [edge for edge in index.ordered_edges if edge.v != index.root]
    if any(edge.arrival <= edge.start for edge in edges):
        return lazy_shortest_path_tree_upper_bound(index)

    # label = (vertex, arrival time, path cost, parent label id, physical eid)
    labels: List[Tuple[str, float, float, Optional[int], str]] = []
    terminal_label: Dict[str, int] = {}
    pending_best: Dict[Tuple[str, float], int] = {}
    best_cost: Dict[str, float] = {index.root: 0.0}
    best_label: Dict[str, Optional[int]] = {index.root: None}
    monotone_arrivals = all(
        left.arrival <= right.arrival
        for left, right in zip(edges, edges[1:])
    )
    pending: Any = deque() if monotone_arrivals else []
    edge_pos = 0

    while edge_pos < len(edges) or pending:
        next_start = edges[edge_pos].start if edge_pos < len(edges) else math.inf
        while pending and pending[0][0] <= next_start:
            arrival, label_id, vertex = (
                pending.popleft() if monotone_arrivals else heapq.heappop(pending)
            )
            key = (vertex, arrival)
            if pending_best.get(key) != label_id:
                continue
            del pending_best[key]
            label = labels[label_id]
            if label[2] < best_cost.get(vertex, math.inf):
                best_cost[vertex] = label[2]
                best_label[vertex] = label_id

        if edge_pos == len(edges):
            break
        edge = edges[edge_pos]
        edge_pos += 1
        source_cost = best_cost.get(edge.u, math.inf)
        if source_cost == math.inf:
            continue
        candidate = source_cost + edge.weight
        key = (edge.v, edge.arrival)
        old_label_id = pending_best.get(key)
        if old_label_id is not None and labels[old_label_id][2] <= candidate:
            continue
        parent_label = best_label.get(edge.u)
        label_id = len(labels)
        labels.append((edge.v, edge.arrival, candidate, parent_label, edge.eid))
        pending_best[key] = label_id
        old_terminal_label = terminal_label.get(edge.v)
        if (
            old_terminal_label is None
            or candidate < labels[old_terminal_label][2]
        ):
            terminal_label[edge.v] = label_id
        item = (edge.arrival, label_id, edge.v)
        if monotone_arrivals:
            pending.append(item)
        else:
            heapq.heappush(pending, item)

    union: List[str] = []
    union_set: Set[str] = set()
    expanded_labels: Set[int] = set()
    for vertex in sorted(index.reachable - {index.root}):
        label_id = terminal_label.get(vertex)
        if label_id is None:
            return math.inf, []
        while label_id is not None and label_id not in expanded_labels:
            expanded_labels.add(label_id)
            _, _, _, parent_label, edge_id = labels[label_id]
            if edge_id not in union_set:
                union_set.add(edge_id)
                union.append(edge_id)
            label_id = parent_label

    tree = _tree_from_edge_ids(graph, index.root, union, index.reachable)
    if len(tree) != len(index.reachable) - 1:
        return math.inf, []
    return sum(edge.weight for edge in tree), [edge.eid for edge in tree]


def _transitive_closure_cost(dag: TimeStateDAG) -> Dict[str, Dict[str, float]]:
    """Return all-pairs shortest-path costs by repeated DAG dynamic programming."""
    nodes = [dag.source] + list(dag.states) + list(dag.terminals.values())
    result: Dict[str, Dict[str, float]] = {}
    order = dag.topological_order()
    for source in nodes:
        dist = {u: math.inf for u in nodes}
        dist[source] = 0.0
        # If source is not the global source, process it and all successors in
        # topological order.  Nodes before source simply remain unreachable.
        for u in order:
            if dist[u] == math.inf:
                continue
            for v, cost, _ in dag.adj.get(u, []):
                if dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
        result[source] = dist
    return result


def _cheapest_path(dag: object, target: str) -> Tuple[float, List[str]]:
    """Find one cheapest root-to-target path and return its physical edge IDs."""
    dist: Dict[str, float] = {dag.source: 0.0}
    previous: Dict[str, Tuple[str, Optional[str]]] = {}
    heap: List[Tuple[float, str]] = [(0.0, dag.source)]
    while heap:
        cost, u = heapq.heappop(heap)
        if cost != dist.get(u, math.inf):
            continue
        if u == target:
            break
        for v, edge_cost, physical in dag.adj.get(u, []):
            candidate = cost + edge_cost
            if candidate < dist.get(v, math.inf):
                dist[v] = candidate
                previous[v] = (u, physical)
                heapq.heappush(heap, (candidate, v))
    if target not in dist:
        return math.inf, []
    path: List[str] = []
    node = target
    while node != dag.source:
        parent, physical = previous[node]
        if physical is not None:
            path.append(physical)
        node = parent
    path.reverse()
    return dist[target], path


def validate_temporal_tree(
    graph: TemporalGraph,
    root: str,
    chosen: Sequence[TemporalEdge],
    required: Optional[Set[str]] = None,
) -> bool:
    """Check the full rooted temporal-tree definition used by MST_w."""
    required = set(required or graph.vertices)
    if root not in required or len(chosen) != len(required) - 1:
        return False
    if any(e.u not in required or e.v not in required or e.v == root for e in chosen):
        return False
    indegree = {v: 0 for v in required}
    forest: Dict[str, str] = {}
    for e in chosen:
        indegree[e.v] += 1
        if indegree[e.v] != 1 or not _forest_add(forest, e.u, e.v):
            return False
    if indegree[root] != 0 or any(indegree[v] != 1 for v in required if v != root):
        return False
    return temporal_reachable(graph, root, chosen) >= required


def _tree_from_edge_ids(
    graph: TemporalGraph,
    root: str,
    edge_ids: Iterable[str],
    required: Optional[Set[str]] = None,
) -> List[TemporalEdge]:
    """Extract a rooted temporal tree from any temporally root-reachable subgraph."""
    by_id = {e.eid: e for e in graph.edges}
    chosen = [by_id[eid] for eid in dict.fromkeys(edge_ids) if eid in by_id]
    arrival, parent = earliest_arrival_tree(
        TemporalGraph(graph.vertices, chosen), root
    )
    required = set(required or graph.vertices)
    if any(arrival[v] == math.inf for v in required):
        return []
    tree = [parent[v] for v in required if v != root]
    return tree if validate_temporal_tree(graph, root, tree, required) else []


def path_union_upper_bound(dag: object) -> Tuple[float, List[str]]:
    """Build a feasible MST_w upper bound from cheapest root-terminal paths.

    This is deliberately *not* an exact DST solver.  It solves one shortest
    path per terminal, takes the union of physical edges, and then extracts a
    rooted temporal tree.  The resulting cost is a valid upper bound and is a
    useful scalable baseline for validating future DST approximations.
    """
    union: List[str] = []
    for terminal in dag.terminals.values():
        cost, path = _cheapest_path(dag, terminal)
        if cost == math.inf:
            return math.inf, []
        for eid in path:
            if eid not in union:
                union.append(eid)
    tree = _tree_from_edge_ids(dag.graph, dag.root, union, dag.reachable)
    if len(tree) != len(dag.reachable) - 1:
        return math.inf, []
    return sum(e.weight for e in tree), [e.eid for e in tree]


def shortest_path_tree_upper_bound(dag: object) -> Tuple[float, List[str]]:
    """Build a feasible upper bound using one shortest-path tree of a DAG.

    Unlike :func:`path_union_upper_bound`, which repeats Dijkstra once per
    terminal, this performs a single topological relaxation from the root and
    backtracks the resulting predecessor tree.  Every retained physical edge
    lies on a shortest root-to-terminal path, so its union is temporally
    root-reachable; ``_tree_from_edge_ids`` then removes redundant edges.

    This is a linear-time *feasible-solution constructor* after the DAG has
    been materialized.  It is not an exact Directed Steiner Tree solver.
    """
    order = dag.topological_order()
    dist: Dict[str, float] = {dag.source: 0.0}
    previous: Dict[str, Tuple[str, Optional[str]]] = {}
    for u in order:
        base = dist.get(u, math.inf)
        if base == math.inf:
            continue
        for v, edge_cost, physical in dag.adj.get(u, []):
            candidate = base + edge_cost
            if candidate < dist.get(v, math.inf):
                dist[v] = candidate
                previous[v] = (u, physical)

    union: List[str] = []
    for terminal in dag.terminals.values():
        if terminal not in dist:
            return math.inf, []
        node = terminal
        while node != dag.source:
            parent, physical = previous[node]
            if physical is not None and physical not in union:
                union.append(physical)
            node = parent
    tree = _tree_from_edge_ids(dag.graph, dag.root, union, dag.reachable)
    if len(tree) != len(dag.reachable) - 1:
        return math.inf, []
    return sum(e.weight for e in tree), [e.eid for e in tree]


def subset_dp_dst(dag: object) -> Tuple[float, List[str]]:
    """Backward-compatible name for the path-union upper-bound baseline."""
    return path_union_upper_bound(dag)


def _forest_add(parent: Dict[str, str], u: str, v: str) -> bool:
    parent.setdefault(u, u)
    parent.setdefault(v, v)

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    ru, rv = find(u), find(v)
    if ru == rv:
        return False
    parent[ru] = rv
    return True


def temporal_reachable(graph: TemporalGraph, root: str, chosen: Sequence[TemporalEdge]) -> Set[str]:
    arrival = {v: math.inf for v in graph.vertices}
    arrival[root] = -math.inf
    changed = True
    while changed:
        changed = False
        for e in sorted(chosen, key=lambda x: (x.start, x.arrival, x.eid)):
            if arrival[e.u] <= e.start and e.arrival < arrival[e.v]:
                arrival[e.v] = e.arrival
                changed = True
    return {v for v, t in arrival.items() if t < math.inf}


def exact_temporal_tree(graph: TemporalGraph, root: str) -> Tuple[float, List[TemporalEdge]]:
    """Small-instance exact verifier for MST_w.

    It enumerates acyclic edge subsets with pruning.  A selected edge is treated
    as one temporal edge; the result is checked by temporal reachability.
    """
    edges = sorted(graph.edges, key=lambda e: (e.weight, e.arrival, e.eid))
    base_reachable = temporal_reachable(graph, root, graph.edges)
    n = len(base_reachable)
    arrival, initial_parent = earliest_arrival_tree(graph, root)
    if all(arrival[v] < math.inf for v in base_reachable):
        best = [initial_parent[v] for v in base_reachable if v != root]
        best_cost = sum(e.weight for e in best)
    else:
        best_cost = math.inf
        best = []

    def lower_bound(i: int, selected: List[TemporalEdge]) -> float:
        missing = max(0, n - 1 - len(selected))
        available = edges[i :]
        if len(available) < missing:
            return math.inf
        return sum(e.weight for e in available[:missing]) if missing else 0.0

    def dfs(i: int, selected: List[TemporalEdge], cost: float, parent: Dict[str, str]) -> None:
        nonlocal best_cost, best
        if cost + lower_bound(i, selected) >= best_cost:
            return
        reached = temporal_reachable(graph, root, selected)
        if len(reached & base_reachable) == n:
            candidate = list(selected)
            if validate_temporal_tree(graph, root, candidate, base_reachable):
                best_cost, best = cost, candidate
            return
        if i == len(edges) or len(selected) >= n - 1:
            return
        e = edges[i]
        if e.v == root or e.u not in base_reachable or e.v not in base_reachable:
            dfs(i + 1, selected, cost, parent)
            return
        # Include branch only when the underlying graph remains acyclic.
        snapshot = dict(parent)
        if _forest_add(parent, e.u, e.v):
            selected.append(e)
            dfs(i + 1, selected, cost + e.weight, parent)
            selected.pop()
        parent.clear()
        parent.update(snapshot)
        dfs(i + 1, selected, cost, parent)

    dfs(0, [], 0.0, {})
    return best_cost, best


def demo() -> None:
    graph = TemporalGraph(
        ["r", "a", "b", "c"],
        [
            TemporalEdge("e1", "r", "a", 1, 2, 2),
            TemporalEdge("e2", "r", "b", 1, 3, 8),
            TemporalEdge("e3", "a", "b", 2, 4, 1),
            TemporalEdge("e4", "b", "c", 4, 5, 1),
            TemporalEdge("e5", "a", "c", 2, 6, 5),
        ],
    )
    arrival, parent = earliest_arrival_tree(graph, "r")
    print("MST_a arrival:", arrival)
    print("MST_a parent:", {v: e.eid for v, e in parent.items()})
    dag = TimeStateDAG(graph, "r")
    print("DAG summary:", dag.summary())
    dag_bound, dag_tree = path_union_upper_bound(dag)
    print("DAG path-union upper bound:", dag_bound, dag_tree)
    te_dag = TeGraphDAG(graph, "r")
    print("TeGraph summary:", te_dag.summary())
    te_bound, te_tree = path_union_upper_bound(te_dag)
    print("TeGraph path-union upper bound:", te_bound, te_tree)
    lazy = LazyTeGraphIndex(graph, "r")
    lazy_bound, lazy_tree = lazy_shortest_path_tree_upper_bound(lazy)
    print("Lazy hyper-DAG upper bound:", lazy_bound, lazy_tree)
    fast_lazy_bound, fast_lazy_tree = lazy_label_sweep_upper_bound(lazy)
    print("Fast lazy label-sweep upper bound:", fast_lazy_bound, fast_lazy_tree)
    cost, tree = exact_temporal_tree(graph, "r")
    print("MST_w exact:", cost, [e.eid for e in tree])
    assert arrival["c"] == 5
    assert set(dag_tree) == {"e1", "e3", "e4"}
    assert set(te_tree) == {"e1", "e3", "e4"}
    assert lazy_bound == te_bound == 4
    assert set(lazy_tree) == {"e1", "e3", "e4"}
    assert fast_lazy_bound == 4
    assert set(fast_lazy_tree) == {"e1", "e3", "e4"}
    assert cost == 4.0


def self_test() -> None:
    """Run small semantic checks used while evolving the reduction."""
    demo()

    unreachable = TemporalGraph(
        ["r", "a", "x"],
        [TemporalEdge("e", "r", "a", 1, 2, 3)],
    )
    dag = TimeStateDAG(unreachable, "r")
    assert dag.reachable == {"r", "a"}
    assert path_union_upper_bound(dag) == (3, ["e"])
    assert exact_temporal_tree(unreachable, "r")[0] == 3

    equal_time = TemporalGraph(
        ["r", "a", "b"],
        [
            TemporalEdge("e1", "r", "a", 0, 0, 1),
            TemporalEdge("e2", "a", "b", 0, 0, 1),
        ],
    )
    assert temporal_reachable(equal_time, "r", equal_time.edges) == {"r", "a", "b"}
    assert path_union_upper_bound(TimeStateDAG(equal_time, "r"))[0] == 2

    same_time_cycle = TemporalGraph(
        ["r", "a", "b"],
        [
            TemporalEdge("e1", "r", "a", 0, 0, 1),
            TemporalEdge("e2", "a", "b", 0, 0, 1),
            TemporalEdge("e3", "b", "a", 0, 0, 1),
        ],
    )
    try:
        TimeStateDAG(same_time_cycle, "r")
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("expected equal-time state cycle to be rejected")

    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--root", default="r")
    parser.add_argument("--mode", choices=["a", "dag", "exact", "all"], default="all")
    parser.add_argument("--dag-style", choices=["huang", "tegraph"], default="huang")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.demo or not args.input:
        demo()
        return
    graph = TemporalGraph.from_json(args.input)
    if args.mode in {"a", "all"}:
        arrival, parent = earliest_arrival_tree(graph, args.root)
        print(json.dumps({"arrival": arrival, "parent": {v: e.eid for v, e in parent.items()}}, indent=2))
    if args.mode in {"dag", "all"}:
        dag_cls = TeGraphDAG if args.dag_style == "tegraph" else TimeStateDAG
        dag = dag_cls(graph, args.root)
        print(json.dumps({"dag": dag.summary(), "upper_bound": path_union_upper_bound(dag)}, indent=2))
    if args.mode in {"exact", "all"}:
        cost, tree = exact_temporal_tree(graph, args.root)
        print(json.dumps({"cost": cost, "edges": [e.eid for e in tree]}, indent=2))


if __name__ == "__main__":
    main()
