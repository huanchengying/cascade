#!/usr/bin/env python3
"""Compact DAG-based temporal graph computing framework.

The design follows the useful part of TeGraph's architecture:

    temporal stream -> time-aware DAG index -> topological single scan

The implementation deliberately separates the graph/index from applications.
Applications implement ``init``, ``update`` and ``aggregate`` while the same
``DAGSingleScan`` executor is reused for reachability, fastest path, shortest
path and temporal-MST routines.  Unlike the research prototype, the graph
payload is stored in NumPy numeric arrays and per-vertex time channels use CSR
offsets; no Python ``TemporalEdge`` object or adjacency-list tuple is retained
for every input edge.

Semantics
---------
An edge is ``(u, v, start, arrival, weight)`` and a path may use the next edge
when ``previous.arrival <= next.start``.  Input edge order is normalized by
``(start, arrival)``.  The compact builder expects integer timestamps and an
integer positive duration for event files, which is appropriate for the
Stack Overflow trace.  General weighted temporal edges can be loaded through
``from_arrays``.

This is an execution framework and a set of baselines, not a claim that
general weighted MST_w is polynomial: ``mst_w_upper_bound`` is the same
one-shortest-path-per-terminal feasible construction used by the prototype.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import resource
import sys
import time
from array import array
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only on minimal installs
    raise RuntimeError(
        "temporal_dag_framework.py requires NumPy; install it with `python3 -m pip install numpy`"
    ) from exc


I64_MIN = np.iinfo(np.int64).min
I64_MAX = np.iinfo(np.int64).max
I32_MIN = np.iinfo(np.int32).min
I32_MAX = np.iinfo(np.int32).max
F64_INF = np.inf


@dataclass(frozen=True)
class CompactEdge:
    """A short-lived edge view; the graph itself stores only numeric arrays."""

    source: int
    target: int
    start: int
    arrival: int
    weight: float
    edge_id: int


@dataclass
class LabelStore:
    """Compact immutable labels used only when an application needs paths."""

    vertex: array
    arrival: array
    cost: array
    parent: array
    edge: array
    origin: array

    @classmethod
    def empty(cls) -> "LabelStore":
        return cls(
            array("i"),
            array("q"),
            array("d"),
            array("q"),
            array("q"),
            array("q"),
        )

    def add(
        self,
        vertex: int,
        arrival: int,
        cost: float,
        parent: int,
        edge_id: int,
        origin_time: int = I64_MIN,
    ) -> int:
        label_id = len(self.vertex)
        self.vertex.append(vertex)
        self.arrival.append(arrival)
        self.cost.append(cost)
        self.parent.append(parent)
        self.edge.append(edge_id)
        self.origin.append(origin_time)
        return label_id

    def path_edges(self, label_id: int) -> List[int]:
        result: List[int] = []
        while label_id >= 0:
            result.append(int(self.edge[label_id]))
            label_id = int(self.parent[label_id])
        result.reverse()
        return result


@dataclass
class TemporalQueryResult:
    """Result of a single-source temporal application."""

    graph: "CompactTemporalDAG"
    source: int
    values: np.ndarray
    arrival: np.ndarray
    predecessor: Optional[np.ndarray] = None
    labels: Optional[LabelStore] = None
    active_labels: Optional[np.ndarray] = None

    def reachable_vertices(self) -> np.ndarray:
        return np.flatnonzero(self.arrival != self.graph.time_max)

    def path_to(self, target: Union[int, str]) -> List[int]:
        target_id = self.graph.vertex_id(target)
        if self.arrival[target_id] == self.graph.time_max:
            return []
        if self.labels is not None and self.active_labels is not None:
            label_id = int(self.active_labels[target_id])
            return self.labels.path_edges(label_id)
        if self.predecessor is None:
            return []
        path: List[int] = []
        vertex = target_id
        while vertex != self.source:
            edge_id = int(self.predecessor[vertex])
            if edge_id < 0:
                return []
            path.append(edge_id)
            vertex = int(self.graph.src[edge_id])
        path.reverse()
        return path


@dataclass
class CompactTemporalDAG:
    """Integer/CSR representation of a temporal graph's time-state DAG.

    ``src/dst/start`` are physically sorted in topological edge order.
    ``arrival_time`` and ``weight`` are optional arrays: constant values are
    kept as scalar metadata.  ``out_offsets/out_times`` and ``in_offsets/in_times`` encode
    the TeGraph-style endpoint-time channels without materializing waiting
    arcs.  A virtual transition is a binary-search/prefix operation over a
    channel, so the stored graph remains O(|E| + |V|) numeric words.
    """

    src: np.ndarray
    dst: np.ndarray
    start: np.ndarray
    # Constant-duration/constant-weight streams use scalar metadata instead
    # of allocating one more full-width NumPy array per edge.
    arrival_time: Optional[np.ndarray]
    weight: Optional[np.ndarray]
    vertex_count: int
    out_offsets: np.ndarray
    out_times: np.ndarray
    in_offsets: np.ndarray
    in_times: np.ndarray
    vertex_names: Optional[List[Any]] = None
    name_to_id: Optional[Dict[Any, int]] = None
    root_id: Optional[int] = None
    input_sorted: bool = False
    duration_scalar: Optional[int] = None
    weight_scalar: Optional[float] = None

    @classmethod
    def from_event_file(
        cls,
        path: Union[str, Path],
        *,
        duration: int = 1,
        default_weight: float = 1.0,
        max_edges: Optional[int] = None,
        root: Optional[str] = None,
        keep_names: bool = True,
        numeric_ids: Optional[bool] = None,
        numeric_direct_ids: bool = False,
        compress_constants: bool = True,
    ) -> "CompactTemporalDAG":
        """Build a compact graph from ``source destination timestamp [weight]``."""
        if duration <= 0:
            raise ValueError("duration must be positive so the expansion is acyclic")
        if numeric_direct_ids and keep_names:
            raise ValueError("numeric_direct_ids requires keep_names=False for sparse raw IDs")
        source_ids: Dict[Any, int] = {}
        names: List[Any] = []
        numeric_mode = numeric_ids
        direct_numeric = bool(numeric_direct_ids and numeric_mode is not False)
        root_present = False
        direct_root: Optional[int] = None
        src_buf = array("i")
        dst_buf = array("i")
        start_buf = array("i")
        arrival_buf: Optional[array] = None if compress_constants else array("q")
        weight_buf: Optional[array] = None if compress_constants else array("d")
        has_explicit_weights: Optional[bool] = None

        def vertex_id(name: Any) -> int:
            value = source_ids.get(name)
            if value is None:
                value = len(source_ids)
                source_ids[name] = value
                if keep_names:
                    names.append(name)
            return value

        edge_rows = 0
        with Path(path).open("r", encoding="ascii") as handle:
            for line_number, line in enumerate(handle, 1):
                if max_edges is not None and edge_rows >= max_edges:
                    break
                fields = line.split()
                if not fields:
                    continue
                if len(fields) not in (3, 4):
                    raise ValueError(
                        f"line {line_number}: expected source destination timestamp [weight]"
                    )
                row_has_weight = len(fields) == 4
                if has_explicit_weights is None:
                    has_explicit_weights = row_has_weight
                    if compress_constants and row_has_weight:
                        weight_buf = array("d")
                elif row_has_weight != has_explicit_weights:
                    raise ValueError(
                        f"line {line_number}: mixed weighted and unweighted rows are unsupported"
                    )
                if numeric_mode is None:
                    try:
                        int(fields[0])
                        int(fields[1])
                        numeric_mode = True
                    except ValueError:
                        numeric_mode = False
                if numeric_mode and direct_numeric and direct_root is None and root is not None:
                    try:
                        direct_root = int(root)
                    except ValueError as exc:
                        raise ValueError("numeric_direct_ids requires an integer root") from exc
                if numeric_mode:
                    try:
                        u_raw = int(fields[0])
                        v_raw = int(fields[1])
                    except ValueError as exc:
                        raise ValueError(
                            f"line {line_number}: mixed numeric and named vertex IDs"
                        ) from exc
                    if direct_numeric:
                        if u_raw < I32_MIN or u_raw > I32_MAX or v_raw < I32_MIN or v_raw > I32_MAX:
                            raise ValueError("direct numeric vertex IDs must fit int32")
                        u, v = u_raw, v_raw
                        if direct_root is not None and (u_raw == direct_root or v_raw == direct_root):
                            root_present = True
                    else:
                        u = vertex_id(u_raw)
                        v = vertex_id(v_raw)
                else:
                    u = vertex_id(fields[0])
                    v = vertex_id(fields[1])
                timestamp = int(fields[2])
                weight = float(fields[3]) if len(fields) == 4 else default_weight
                src_buf.append(u)
                dst_buf.append(v)
                try:
                    start_buf.append(timestamp)
                except OverflowError:
                    widened_start = array("q", start_buf)
                    widened_start.append(timestamp)
                    start_buf = widened_start
                if arrival_buf is not None:
                    arrival_buf.append(timestamp + duration)
                if weight_buf is not None:
                    weight_buf.append(weight)
                edge_rows += 1

        src_a = np.asarray(src_buf, dtype=np.int32)
        dst_a = np.asarray(dst_buf, dtype=np.int32)
        if direct_numeric and numeric_mode:
            if len(src_a):
                max_vertex = int(max(src_a.max(), dst_a.max()))
                min_vertex = int(min(src_a.min(), dst_a.min()))
                if min_vertex < 0:
                    raise ValueError("direct numeric vertex IDs must be nonnegative")
                direct_vertex_count = max_vertex + 1
            else:
                direct_vertex_count = 0
            root_id = direct_root if root is not None and root_present else None
            if root is not None and not root_present:
                raise ValueError(f"root vertex {root!r} does not occur in input prefix")
            names_arg = None
            name_map_arg = None
            vertex_count_arg = direct_vertex_count
        else:
            if root is None:
                root_id = None
            else:
                root_key: Any = int(root) if numeric_mode else root
                root_id = source_ids.get(root_key)
                if root_id is None:
                    raise ValueError(f"root vertex {root!r} does not occur in input prefix")
            names_arg = names if keep_names else None
            name_map_arg = source_ids if keep_names else None
            vertex_count_arg = len(source_ids)
        start_a = np.asarray(start_buf, dtype=np.int64)
        arrival_a = None if arrival_buf is None else np.asarray(arrival_buf, dtype=np.int64)
        weight_a = None if weight_buf is None else np.asarray(weight_buf, dtype=np.float64)
        del src_buf, dst_buf, start_buf, arrival_buf, weight_buf
        if not keep_names:
            del source_ids, names
        return cls.from_arrays(
            src_a,
            dst_a,
            start_a,
            arrival_a,
            weight_a,
            vertex_count=vertex_count_arg,
            vertex_names=names_arg,
            name_to_id=name_map_arg,
            root_id=root_id,
            input_sorted=False,
            compress_constants=compress_constants,
            duration_scalar=duration if arrival_a is None else None,
            weight_scalar=default_weight if weight_a is None else None,
        )

    @classmethod
    def from_arrays(
        cls,
        src: Union[Sequence[int], np.ndarray],
        dst: Union[Sequence[int], np.ndarray],
        start: Union[Sequence[int], np.ndarray],
        arrival: Optional[Union[Sequence[int], np.ndarray]],
        weight: Optional[Union[Sequence[float], np.ndarray]],
        *,
        vertex_count: Optional[int] = None,
        vertex_names: Optional[List[Any]] = None,
        name_to_id: Optional[Dict[Any, int]] = None,
        root_id: Optional[int] = None,
        sort_edges: bool = True,
        input_sorted: bool = False,
        compress_constants: bool = True,
        duration_scalar: Optional[int] = None,
        weight_scalar: Optional[float] = None,
    ) -> "CompactTemporalDAG":
        """Construct from numeric edge arrays and build CSR time channels."""
        src_a = np.asarray(src, dtype=np.int32)
        dst_a = np.asarray(dst, dtype=np.int32)
        start_a = np.asarray(start)
        if start_a.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
            start_a = np.asarray(start, dtype=np.int64)
        arrival_a = None if arrival is None else np.asarray(arrival)
        if arrival_a is not None and arrival_a.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
            arrival_a = np.asarray(arrival, dtype=np.int64)
        weight_a = None if weight is None else np.asarray(weight, dtype=np.float64)
        lengths = {len(src_a), len(dst_a), len(start_a)}
        if arrival_a is not None:
            lengths.add(len(arrival_a))
        if weight_a is not None:
            lengths.add(len(weight_a))
        if len(lengths) != 1:
            raise ValueError("edge arrays must have equal length")
        if arrival_a is None and (duration_scalar is None or duration_scalar <= 0):
            raise ValueError("duration_scalar must be positive when arrival is omitted")
        if weight_a is None and weight_scalar is None:
            raise ValueError("weight_scalar is required when weight is omitted")
        if arrival_a is not None and np.any(arrival_a <= start_a):
            raise ValueError("arrival must be > start so the expansion is a DAG")
        if vertex_count is None:
            vertex_count = int(max(src_a.max(initial=-1), dst_a.max(initial=-1)) + 1)
        if len(src_a) and (
            int(src_a.min()) < 0
            or int(dst_a.min()) < 0
            or int(src_a.max()) >= vertex_count
            or int(dst_a.max()) >= vertex_count
        ):
            raise ValueError("edge endpoint exceeds vertex_count")
        if sort_edges and len(src_a) > 1:
            order = (
                np.argsort(start_a, kind="stable")
                if arrival_a is None
                else np.lexsort((arrival_a, start_a))
            )
            src_a = src_a[order]
            dst_a = dst_a[order]
            start_a = start_a[order]
            if arrival_a is not None:
                arrival_a = arrival_a[order]
            if weight_a is not None:
                weight_a = weight_a[order]
            input_sorted = True
        # Most temporal traces use Unix-second timestamps, which fit in
        # signed int32.  Keep int64 automatically when a caller supplies a
        # wider range so the representation remains lossless.
        if len(start_a) == 0 or (
            int(start_a.min()) > I32_MIN
            and int(start_a.max()) < I32_MAX
            and (
                (
                    int(start_a.min()) + int(duration_scalar) > I32_MIN
                    and int(start_a.max()) + int(duration_scalar) < I32_MAX
                )
                if arrival_a is None
                else (
                    int(arrival_a.min(initial=0)) > I32_MIN
                    and int(arrival_a.max(initial=0)) < I32_MAX
                )
            )
        ):
            start_a = start_a.astype(np.int32, copy=False)
            if arrival_a is not None:
                arrival_a = arrival_a.astype(np.int32, copy=False)
        out_offsets, out_times = _build_channel(src_a, start_a, vertex_count)
        arrival_for_channel = (
            start_a + np.asarray(duration_scalar, dtype=start_a.dtype)
            if arrival_a is None
            else arrival_a
        )
        in_offsets, in_times = _build_channel(dst_a, arrival_for_channel, vertex_count)
        arrival_store: Optional[np.ndarray] = arrival_a
        weight_store: Optional[np.ndarray] = weight_a
        if compress_constants and len(start_a) and arrival_a is not None:
            duration = int(arrival_a[0] - start_a[0])
            if np.all(arrival_a - start_a == duration):
                duration_scalar = duration
                arrival_store = None
        if compress_constants and len(start_a) and weight_a is not None:
            weight_value = float(weight_a[0])
            if np.all(weight_a == weight_value):
                weight_scalar = weight_value
                weight_store = None
        return cls(
            src_a,
            dst_a,
            start_a,
            arrival_store,
            weight_store,
            int(vertex_count),
            out_offsets,
            out_times,
            in_offsets,
            in_times,
            vertex_names,
            name_to_id,
            root_id,
            input_sorted,
            duration_scalar,
            weight_scalar,
        )

    @property
    def edge_count(self) -> int:
        return int(self.src.size)

    @property
    def state_count(self) -> int:
        return int(self.out_times.size + self.in_times.size)

    @property
    def time_dtype(self) -> np.dtype:
        return self.start.dtype

    @property
    def time_min(self) -> int:
        return int(np.iinfo(self.time_dtype).min)

    @property
    def time_max(self) -> int:
        return int(np.iinfo(self.time_dtype).max)

    @property
    def index_dtype(self) -> np.dtype:
        return np.dtype(np.int32 if self.edge_count <= I32_MAX else np.int64)

    @property
    def arrival_monotone(self) -> bool:
        if self.arrival_time is None:
            return bool(self.edge_count < 2 or np.all(self.start[:-1] <= self.start[1:]))
        return bool(self.edge_count < 2 or np.all(self.arrival_time[:-1] <= self.arrival_time[1:]))

    def arrival_at(self, edge_id: int) -> int:
        if self.arrival_time is None:
            return int(self.start[edge_id] + int(self.duration_scalar))
        return int(self.arrival_time[edge_id])

    def weight_at(self, edge_id: int) -> float:
        if self.weight is None:
            return float(self.weight_scalar)
        return float(self.weight[edge_id])

    def edge_weights(self, edge_ids: Sequence[int]) -> np.ndarray:
        ids = np.asarray(edge_ids, dtype=np.int64)
        if self.weight is None:
            return np.full(ids.shape, float(self.weight_scalar), dtype=np.float64)
        return self.weight[ids]

    def vertex_id(self, vertex: Union[int, str]) -> int:
        if isinstance(vertex, int):
            if vertex < 0 or vertex >= self.vertex_count:
                raise ValueError(f"unknown vertex id: {vertex}")
            return vertex
        if self.name_to_id is None:
            try:
                numeric_vertex = int(vertex)
            except (TypeError, ValueError):
                numeric_vertex = None
            if numeric_vertex is not None and 0 <= numeric_vertex < self.vertex_count:
                return numeric_vertex
        if self.name_to_id is None or vertex not in self.name_to_id:
            raise ValueError(f"vertex names were discarded or vertex is unknown: {vertex!r}")
        return self.name_to_id[vertex]

    def vertex_name(self, vertex: int) -> Union[str, int]:
        if self.vertex_names is None:
            return vertex
        return self.vertex_names[vertex]

    def edge_view(self, edge_id: int) -> CompactEdge:
        return CompactEdge(
            int(self.src[edge_id]),
            int(self.dst[edge_id]),
            int(self.start[edge_id]),
            self.arrival_at(edge_id),
            self.weight_at(edge_id),
            edge_id,
        )

    def numeric_memory_bytes(self) -> int:
        arrays = (
            self.src,
            self.dst,
            self.start,
            self.arrival_time,
            self.weight,
            self.out_offsets,
            self.out_times,
            self.in_offsets,
            self.in_times,
        )
        return int(sum(array_item.nbytes for array_item in arrays if array_item is not None))

    def summary(self) -> Dict[str, Any]:
        out_chain_arcs = int(
            np.sum(np.maximum(0, self.out_offsets[1:] - self.out_offsets[:-1] - 1))
        )
        in_counts = np.diff(self.in_offsets)
        out_nonempty = self.out_offsets[1:] > self.out_offsets[:-1]
        last_out = np.full(self.vertex_count, self.time_min, dtype=self.out_times.dtype)
        last_out[out_nonempty] = self.out_times[self.out_offsets[1:][out_nonempty] - 1]
        in_threshold = np.repeat(last_out, in_counts)
        in_wait_arcs = int(np.count_nonzero(self.in_times <= in_threshold))
        return {
            "vertices": self.vertex_count,
            "edges": self.edge_count,
            "states": self.state_count,
            "out_states": int(self.out_times.size),
            "in_states": int(self.in_times.size),
            "numeric_memory_bytes": self.numeric_memory_bytes(),
            "numeric_memory_mib": self.numeric_memory_bytes() / (1024 * 1024),
            "arrival_monotone": self.arrival_monotone,
            "input_sorted": self.input_sorted,
            "duration_scalar": self.duration_scalar,
            "weight_scalar": self.weight_scalar,
            "implicit_wait_channel_count": int(np.count_nonzero(self.out_offsets[1:] > self.out_offsets[:-1])),
            "estimated_wait_arcs_not_materialized": out_chain_arcs + in_wait_arcs,
            "out_chain_wait_arcs": out_chain_arcs,
            "in_to_out_wait_arcs": in_wait_arcs,
        }


def _build_channel(vertices: np.ndarray, times: np.ndarray, vertex_count: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0:
        offset_dtype = np.int32 if vertex_count <= I32_MAX else np.int64
        return np.zeros(vertex_count + 1, dtype=offset_dtype), np.empty(0, dtype=times.dtype)
    order = np.lexsort((times, vertices))
    sorted_vertices = vertices[order]
    sorted_times = times[order]
    keep = np.empty(len(order), dtype=bool)
    keep[0] = True
    keep[1:] = (sorted_vertices[1:] != sorted_vertices[:-1]) | (sorted_times[1:] != sorted_times[:-1])
    unique_vertices = sorted_vertices[keep]
    unique_times = sorted_times[keep]
    counts = np.bincount(unique_vertices, minlength=vertex_count)
    offset_dtype = np.int32 if len(unique_times) <= I32_MAX else np.int64
    offsets = np.empty(vertex_count + 1, dtype=offset_dtype)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, unique_times


class TemporalApplication:
    """TeGraph-style application contract for the compact single scan."""

    def init(self, graph: CompactTemporalDAG, source: int, source_time: int) -> Any:
        raise NotImplementedError

    def update(self, state: Any, graph: CompactTemporalDAG, edge_id: int) -> None:
        raise NotImplementedError

    def compare(self, candidate: Any, current: Any) -> bool:
        return candidate < current

    def aggregate(self, state: Any, graph: CompactTemporalDAG, source: int) -> TemporalQueryResult:
        raise NotImplementedError


class DAGSingleScan:
    """Shared topological executor.  The graph supplies the topological order."""

    def run(
        self,
        graph: CompactTemporalDAG,
        application: TemporalApplication,
        source: Union[int, str],
        *,
        source_time: int = 0,
    ) -> TemporalQueryResult:
        source_id = graph.vertex_id(source)
        optimized = getattr(application, "_run_optimized", None)
        if optimized is not None:
            return optimized(graph, source_id, source_time)
        state = application.init(graph, source_id, source_time)
        for edge_id in range(graph.edge_count):
            application.update(state, graph, edge_id)
        return application.aggregate(state, graph, source_id)


class ReachabilityApplication(TemporalApplication):
    """Earliest-arrival reachability; also the Huang ``MST_a`` baseline."""

    @staticmethod
    def _run_optimized(
        graph: CompactTemporalDAG, source: int, source_time: int
    ) -> TemporalQueryResult:
        """Array-local scan used by the built-in application.

        The public application callbacks remain available for custom kernels;
        this path removes one Python method dispatch and one scalar helper call
        per physical edge for the most common MST_a/reachability workload.
        """
        arrival = np.full(graph.vertex_count, graph.time_max, dtype=graph.time_dtype)
        predecessor = np.full(graph.vertex_count, -1, dtype=graph.index_dtype)
        arrival[source] = source_time
        src, dst, starts = graph.src, graph.dst, graph.start
        duration = graph.duration_scalar
        edge_arrivals = graph.arrival_time
        if edge_arrivals is None:
            for edge_id in range(graph.edge_count):
                u = int(src[edge_id])
                start = int(starts[edge_id])
                if arrival[u] <= start:
                    v = int(dst[edge_id])
                    edge_arrival = start + int(duration)
                    if edge_arrival < arrival[v]:
                        arrival[v] = edge_arrival
                        predecessor[v] = edge_id
        else:
            for edge_id in range(graph.edge_count):
                u = int(src[edge_id])
                if arrival[u] <= starts[edge_id]:
                    v = int(dst[edge_id])
                    edge_arrival = edge_arrivals[edge_id]
                    if edge_arrival < arrival[v]:
                        arrival[v] = edge_arrival
                        predecessor[v] = edge_id
        values = arrival.astype(np.float64, copy=True)
        values[arrival == graph.time_max] = np.inf
        return TemporalQueryResult(graph, source, values, arrival, predecessor)

    def init(self, graph: CompactTemporalDAG, source: int, source_time: int) -> Dict[str, Any]:
        arrival = np.full(graph.vertex_count, graph.time_max, dtype=graph.time_dtype)
        predecessor = np.full(graph.vertex_count, -1, dtype=graph.index_dtype)
        arrival[source] = source_time
        return {"arrival": arrival, "predecessor": predecessor, "source_time": source_time}

    def update(self, state: Dict[str, Any], graph: CompactTemporalDAG, edge_id: int) -> None:
        u = int(graph.src[edge_id])
        v = int(graph.dst[edge_id])
        edge_arrival = graph.arrival_at(edge_id)
        if state["arrival"][u] <= graph.start[edge_id] and edge_arrival < state["arrival"][v]:
            state["arrival"][v] = edge_arrival
            state["predecessor"][v] = edge_id

    def aggregate(self, state: Dict[str, Any], graph: CompactTemporalDAG, source: int) -> TemporalQueryResult:
        values = state["arrival"].astype(np.float64, copy=True)
        values[state["arrival"] == graph.time_max] = np.inf
        return TemporalQueryResult(graph, source, values, state["arrival"], state["predecessor"])


class FastestPathApplication(TemporalApplication):
    """Minimum ``last arrival - first departure`` temporal paths.

    ``source_time`` is the earliest allowed departure.  A source edge may
    start later than that bound, so the path's first departure is the edge
    start, matching TeGraph's temporal fastest-path target.
    """

    def __init__(self, return_paths: bool = False) -> None:
        self.return_paths = return_paths

    def init(self, graph: CompactTemporalDAG, source: int, source_time: int) -> Dict[str, Any]:
        duration = np.full(graph.vertex_count, F64_INF, dtype=np.float64)
        arrival = np.full(graph.vertex_count, graph.time_max, dtype=graph.time_dtype)
        active_departure = np.full(graph.vertex_count, graph.time_min, dtype=graph.time_dtype)
        active_label = np.full(graph.vertex_count, -1, dtype=graph.index_dtype)
        terminal_label = np.full(graph.vertex_count, -1, dtype=graph.index_dtype)
        labels = LabelStore.empty() if self.return_paths else None
        duration[source] = 0.0
        arrival[source] = source_time
        return {
            "duration": duration,
            "arrival": arrival,
            "active_departure": active_departure,
            "active_label": active_label,
            "terminal_label": terminal_label,
            "labels": labels,
            "pending": deque() if graph.arrival_monotone else [],
            "pending_departure": np.full(graph.edge_count, graph.time_min, dtype=graph.time_dtype),
            "pending_tail_departure": np.full(graph.vertex_count, graph.time_min, dtype=graph.time_dtype),
            "pending_tail_arrival": np.full(graph.vertex_count, graph.time_max, dtype=graph.time_dtype),
            "pending_tail_token": np.full(graph.vertex_count, -1, dtype=graph.index_dtype),
            "source": source,
            "source_time": source_time,
            "monotone": graph.arrival_monotone,
        }

    @staticmethod
    def _push(state: Dict[str, Any], item: Tuple[int, int, int]) -> None:
        if state["monotone"]:
            state["pending"].append(item)
        else:
            heapq.heappush(state["pending"], item)

    @staticmethod
    def _pop(state: Dict[str, Any]) -> Tuple[int, int, int]:
        return state["pending"].popleft() if state["monotone"] else heapq.heappop(state["pending"])

    def _activate(self, state: Dict[str, Any], edge_start: int) -> None:
        while state["pending"] and state["pending"][0][0] <= edge_start:
            pending_arrival, token, vertex = self._pop(state)
            if state["monotone"]:
                tail_token = int(state["pending_tail_token"][vertex])
                tail_arrival = int(state["pending_tail_arrival"][vertex])
                if pending_arrival == tail_arrival and token != tail_token:
                    continue
                if token == tail_token:
                    state["pending_tail_token"][vertex] = -1
            if state["labels"] is None:
                departure = int(state["pending_departure"][token])
                label_id = -1
            else:
                label_id = token
                departure = int(state["labels"].origin[label_id])
            if departure > state["active_departure"][vertex]:
                state["active_departure"][vertex] = departure
                state["active_label"][vertex] = label_id

    def update(self, state: Dict[str, Any], graph: CompactTemporalDAG, edge_id: int) -> None:
        start = int(graph.start[edge_id])
        self._activate(state, start)
        if start < state["source_time"]:
            return
        u = int(graph.src[edge_id])
        v = int(graph.dst[edge_id])
        if u == state["source"]:
            departure = start
            parent_label = -1
        else:
            departure = int(state["active_departure"][u])
            if departure == graph.time_min:
                return
            parent_label = int(state["active_label"][u])
        arrival = graph.arrival_at(edge_id)
        candidate = float(arrival - departure)
        if state["monotone"]:
            if departure <= state["active_departure"][v]:
                return
            tail_token = int(state["pending_tail_token"][v])
            if tail_token >= 0 and departure <= state["pending_tail_departure"][v]:
                return
        if state["labels"] is None:
            token = edge_id
            state["pending_departure"][edge_id] = departure
        else:
            token = state["labels"].add(
                v,
                arrival,
                candidate,
                parent_label,
                edge_id,
                departure,
            )
        if candidate < state["duration"][v]:
            state["duration"][v] = candidate
            state["arrival"][v] = arrival
            state["terminal_label"][v] = token if state["labels"] is not None else -1
        if state["monotone"]:
            state["pending_tail_departure"][v] = departure
            state["pending_tail_arrival"][v] = arrival
            state["pending_tail_token"][v] = token
        self._push(state, (arrival, token, v))

    def aggregate(self, state: Dict[str, Any], graph: CompactTemporalDAG, source: int) -> TemporalQueryResult:
        self._activate(state, graph.time_max)
        return TemporalQueryResult(
            graph,
            source,
            state["duration"],
            state["arrival"],
            None,
            state["labels"],
            state["terminal_label"] if state["labels"] is not None else None,
        )


class ShortestPathApplication(TemporalApplication):
    """Minimum total edge weight under temporal order.

    The active value of a vertex is the minimum cost among labels whose arrival
    has already passed the current edge start.  Arriving labels are kept in a
    min-heap for arbitrary durations; when arrivals are monotone, a deque is
    used instead.  ``return_paths=True`` adds compact immutable labels, which
    costs O(number of successful relaxations) numeric words only when requested.
    """

    def __init__(self, return_paths: bool = False) -> None:
        self.return_paths = return_paths

    def init(self, graph: CompactTemporalDAG, source: int, source_time: int) -> Dict[str, Any]:
        cost = np.full(graph.vertex_count, F64_INF, dtype=np.float64)
        arrival = np.full(graph.vertex_count, graph.time_max, dtype=graph.time_dtype)
        predecessor = np.full(graph.vertex_count, -1, dtype=graph.index_dtype)
        active_labels = np.full(graph.vertex_count, -1, dtype=graph.index_dtype)
        cost[source] = 0.0
        arrival[source] = source_time
        labels = LabelStore.empty() if self.return_paths else None
        pending: Any = deque() if graph.arrival_monotone else []
        return {
            "cost": cost,
            "arrival": arrival,
            "predecessor": predecessor,
            "active_labels": active_labels,
            "labels": labels,
            "pending": pending,
            "pending_costs": np.full(graph.edge_count, F64_INF, dtype=np.float64),
            "pending_tail_cost": np.full(graph.vertex_count, F64_INF, dtype=np.float64),
            "pending_tail_arrival": np.full(graph.vertex_count, graph.time_max, dtype=graph.time_dtype),
            "pending_tail_token": np.full(graph.vertex_count, -1, dtype=graph.index_dtype),
            "source_time": source_time,
            "monotone": graph.arrival_monotone,
        }

    @staticmethod
    def _push_pending(state: Dict[str, Any], item: Tuple[int, int, int]) -> None:
        if state["monotone"]:
            state["pending"].append(item)
        else:
            heapq.heappush(state["pending"], item)

    @staticmethod
    def _pop_pending(state: Dict[str, Any]) -> Tuple[int, int, int]:
        return state["pending"].popleft() if state["monotone"] else heapq.heappop(state["pending"])

    def _activate(self, state: Dict[str, Any], graph: CompactTemporalDAG, edge_start: int) -> None:
        pending = state["pending"]
        while pending and pending[0][0] <= edge_start:
            arrival, label_or_edge, vertex = self._pop_pending(state)
            if state["monotone"]:
                tail_token = int(state["pending_tail_token"][vertex])
                tail_arrival = int(state["pending_tail_arrival"][vertex])
                if arrival == tail_arrival and label_or_edge != tail_token:
                    continue
                if label_or_edge == tail_token:
                    state["pending_tail_token"][vertex] = -1
                    state["pending_tail_arrival"][vertex] = graph.time_max
                    state["pending_tail_cost"][vertex] = np.inf
            if state["labels"] is None:
                candidate_cost = state["pending_costs"][label_or_edge]
                candidate_edge = label_or_edge
                if candidate_cost < state["cost"][vertex]:
                    state["cost"][vertex] = candidate_cost
                    state["arrival"][vertex] = arrival
                    state["predecessor"][vertex] = candidate_edge
            else:
                label_id = label_or_edge
                candidate_cost = state["labels"].cost[label_id]
                if candidate_cost < state["cost"][vertex]:
                    state["cost"][vertex] = candidate_cost
                    state["arrival"][vertex] = arrival
                    state["active_labels"][vertex] = label_id
                    state["predecessor"][vertex] = state["labels"].edge[label_id]

    def update(self, state: Dict[str, Any], graph: CompactTemporalDAG, edge_id: int) -> None:
        start = int(graph.start[edge_id])
        self._activate(state, graph, start)
        if start < state["source_time"]:
            return
        u = int(graph.src[edge_id])
        v = int(graph.dst[edge_id])
        if state["cost"][u] == np.inf:
            return
        candidate = float(state["cost"][u] + graph.weight_at(edge_id))
        arrival = graph.arrival_at(edge_id)
        if state["monotone"]:
            if candidate >= state["cost"][v]:
                return
            tail_token = int(state["pending_tail_token"][v])
            if tail_token >= 0 and candidate >= state["pending_tail_cost"][v]:
                return
        if state["labels"] is None:
            state["pending_costs"][edge_id] = candidate
            token = edge_id
        else:
            parent_label = int(state["active_labels"][u])
            label_id = state["labels"].add(
                v,
                arrival,
                candidate,
                parent_label,
                edge_id,
            )
            token = label_id
        if state["monotone"]:
            state["pending_tail_cost"][v] = candidate
            state["pending_tail_arrival"][v] = arrival
            state["pending_tail_token"][v] = token
        self._push_pending(state, (arrival, token, v))

    def aggregate(self, state: Dict[str, Any], graph: CompactTemporalDAG, source: int) -> TemporalQueryResult:
        self._activate(state, graph, graph.time_max)
        values = state["cost"].copy()
        return TemporalQueryResult(
            graph,
            source,
            values,
            state["arrival"],
            state["predecessor"] if state["labels"] is not None else None,
            state["labels"],
            state["active_labels"],
        )


def mst_a(graph: CompactTemporalDAG, source: Union[int, str], *, source_time: int = 0) -> TemporalQueryResult:
    """Huang MST_a: earliest-arrival temporal parent tree."""
    return DAGSingleScan().run(graph, ReachabilityApplication(), source, source_time=source_time)


def fastest_path(
    graph: CompactTemporalDAG,
    source: Union[int, str],
    *,
    departure_time: int = 0,
    return_paths: bool = False,
) -> TemporalQueryResult:
    return DAGSingleScan().run(
        graph,
        FastestPathApplication(return_paths=return_paths),
        source,
        source_time=departure_time,
    )


def shortest_path(
    graph: CompactTemporalDAG,
    source: Union[int, str],
    *,
    source_time: int = 0,
    return_paths: bool = False,
) -> TemporalQueryResult:
    return DAGSingleScan().run(
        graph,
        ShortestPathApplication(return_paths=return_paths),
        source,
        source_time=source_time,
    )


def mst_w_upper_bound(
    graph: CompactTemporalDAG,
    source: Union[int, str],
    *,
    source_time: int = 0,
) -> Tuple[float, List[int], TemporalQueryResult]:
    """Feasible MST_w upper bound from one shortest path per reachable vertex."""
    source_id = graph.vertex_id(source)
    paths = shortest_path(graph, source_id, source_time=source_time, return_paths=True)
    reachable = paths.reachable_vertices()
    if len(reachable) == 0 or source_id not in reachable:
        return math.inf, [], paths
    union = set()
    for vertex in reachable:
        if int(vertex) != source_id:
            union.update(paths.path_to(int(vertex)))

    # Extract an actual temporal parent tree from the physical-edge union.
    earliest = np.full(graph.vertex_count, graph.time_max, dtype=graph.time_dtype)
    parent = np.full(graph.vertex_count, -1, dtype=graph.index_dtype)
    earliest[source_id] = source_time
    for edge_id in range(graph.edge_count):
        if edge_id not in union:
            continue
        u = int(graph.src[edge_id])
        v = int(graph.dst[edge_id])
        edge_arrival = graph.arrival_at(edge_id)
        if earliest[u] <= graph.start[edge_id] and edge_arrival < earliest[v]:
            earliest[v] = edge_arrival
            parent[v] = edge_id
    tree = [int(parent[v]) for v in reachable if int(v) != source_id and parent[v] >= 0]
    if len(tree) != len(reachable) - 1:
        return math.inf, [], paths
    return float(np.sum(graph.edge_weights(tree))), tree, paths


def peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def self_test() -> None:
    graph = CompactTemporalDAG.from_arrays(
        [0, 0, 1, 2, 1],
        [1, 2, 2, 3, 3],
        [1, 1, 2, 4, 2],
        [2, 3, 4, 5, 6],
        [2, 8, 1, 1, 5],
        vertex_count=4,
        vertex_names=["r", "a", "b", "c"],
        name_to_id={"r": 0, "a": 1, "b": 2, "c": 3},
    )
    reach = mst_a(graph, "r", source_time=0)
    assert set(reach.reachable_vertices()) == {0, 1, 2, 3}
    assert int(reach.arrival[3]) == 5
    fast = fastest_path(graph, "r", departure_time=0, return_paths=True)
    assert int(fast.values[3]) == 4
    assert fast.path_to("c")
    short = shortest_path(graph, "r", source_time=0, return_paths=True)
    assert short.values[3] == 4
    assert short.path_to("c")
    cost, tree, _ = mst_w_upper_bound(graph, "r", source_time=0)
    assert cost == 4
    assert len(tree) == 3
    assert graph.state_count == 8
    print("compact framework self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--root", default="9")
    parser.add_argument("--duration", type=int, default=1)
    parser.add_argument("--max-edges", type=int)
    parser.add_argument("--application", choices=["reachability", "fastest", "shortest", "mst_a", "mst_w"], default="shortest")
    parser.add_argument("--source-time", type=int, default=0)
    parser.add_argument("--keep-names", action="store_true")
    parser.add_argument(
        "--numeric-direct-ids",
        action="store_true",
        help="for numeric input, use raw nonnegative int32 IDs and skip the vertex dictionary",
    )
    parser.add_argument("--no-compress-constants", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.input:
        parser.error("--input is required unless --self-test is used")
    started = time.perf_counter()
    graph = CompactTemporalDAG.from_event_file(
        args.input,
        duration=args.duration,
        max_edges=args.max_edges,
        root=args.root,
        keep_names=args.keep_names,
        numeric_direct_ids=args.numeric_direct_ids,
        compress_constants=not args.no_compress_constants,
    )
    build_seconds = time.perf_counter() - started
    source = graph.root_id if graph.root_id is not None else graph.vertex_id(args.root)
    result: Any
    solve_started = time.perf_counter()
    if args.application == "reachability" or args.application == "mst_a":
        result = mst_a(graph, source, source_time=args.source_time)
    elif args.application == "fastest":
        result = fastest_path(graph, source, departure_time=args.source_time)
    elif args.application == "shortest":
        result = shortest_path(graph, source, source_time=args.source_time, return_paths=False)
    else:
        result = mst_w_upper_bound(graph, source, source_time=args.source_time)
    solve_seconds = time.perf_counter() - solve_started
    if args.application == "mst_w":
        cost, tree, query = result
        payload = {"cost": cost, "tree_edges": len(tree), "reachable": len(query.reachable_vertices())}
    else:
        payload = {"reachable": len(result.reachable_vertices()), "max_value": float(np.max(result.values[result.values != np.inf]))}
    print(json.dumps({
        "graph": graph.summary(),
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "end_to_end_seconds": build_seconds + solve_seconds,
        "peak_rss_mib": peak_rss_mib(),
        "application": args.application,
        "result": payload,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
