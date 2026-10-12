from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from temporal_mst_prototype import TemporalGraph, earliest_arrival_tree, temporal_reachable, validate_temporal_tree


METHODS = ('path_union', 'ea_branching', 'snapshot_mst', 'star', 'cascade', 'subset_dp')


def earliest_weight_branching(graph, root):
    arrival, _ = earliest_arrival_tree(graph, root)
    parents = {}
    for edge in graph.edges:
        if edge.v == root or edge.u == edge.v:
            continue
        if arrival[edge.u] <= edge.start and edge.arrival == arrival[edge.v]:
            previous = parents.get(edge.v)
            if previous is None or (edge.weight, edge.eid) < (previous.weight, previous.eid):
                parents[edge.v] = edge
    return list(parents.values())


def snapshot_mst(graph, required):
    representative = {vertex: vertex for vertex in required}
    size = {vertex: 1 for vertex in required}

    def find(vertex):
        while representative[vertex] != vertex:
            representative[vertex] = representative[representative[vertex]]
            vertex = representative[vertex]
        return vertex

    selected = []
    for edge in sorted(graph.edges, key=lambda event: (event.weight, event.eid)):
        if edge.u not in required or edge.v not in required:
            continue
        left, right = find(edge.u), find(edge.v)
        if left == right:
            continue
        if size[left] < size[right]:
            left, right = right, left
        representative[right] = left
        size[left] += size[right]
        selected.append(edge)
    return selected


def measured_kernel(graph, root, method, settings):
    import temporal_mst_cascade_star as cascade_module
    import temporal_mst_shared as shared_module
    import temporal_mst_star as star_module

    phases = {}
    calls = {}

    def measured(name, function):
        def invoke(*arguments, **keywords):
            started = time.perf_counter()
            try:
                return function(*arguments, **keywords)
            finally:
                phases[name] = phases.get(name, 0.0) + time.perf_counter() - started
                calls[name] = calls.get(name, 0) + 1
        return invoke

    bindings = (
        ('path_construction', 'shortest_path_union_tree', (shared_module, star_module, cascade_module)),
        ('parent_exchange', '_parent_exchange_descent', (shared_module, star_module, cascade_module)),
        ('conservative_star', '_shared_activation_descent', (star_module, cascade_module)),
        ('candidate_scoring', '_scored_parents', (cascade_module,)),
        ('fixed_parent_dp', '_evaluate_parent', (cascade_module,)),
        ('apply_and_validate', '_apply_cascade_move', (cascade_module,)),
    )
    with ExitStack() as stack:
        for name, attribute, modules in bindings:
            wrapper = measured(name, getattr(modules[0], attribute))
            for module in modules:
                stack.enter_context(patch.object(module, attribute, wrapper))
        started = time.perf_counter()
        if method == 'path_union':
            cost, selected, _ = shared_module.shortest_path_union_tree(graph, root)
            result = {'cost': cost, 'edge_ids': [edge.eid for edge in selected]}
        elif method == 'ea_branching':
            selected = earliest_weight_branching(graph, root)
            result = {'cost': math.fsum(edge.weight for edge in selected), 'edge_ids': [edge.eid for edge in selected]}
        elif method == 'snapshot_mst':
            selected = snapshot_mst(graph, temporal_reachable(graph, root, graph.edges))
            result = {'cost': math.fsum(edge.weight for edge in selected), 'edge_ids': [edge.eid for edge in selected]}
        elif method == 'star':
            result = asdict(star_module.shared_activation_star(graph, root, max_rounds=settings.get('conservative_rounds', 4)))
        elif method == 'cascade':
            result = asdict(cascade_module.cascade_activation_star(graph, root, conservative_rounds=settings.get('conservative_rounds', 4), cascade_rounds=settings.get('cascade_rounds', 2), max_hubs=settings.get('max_hubs', 64), max_parent_evaluations=settings.get('max_parent_evaluations', 128)))
        elif method == 'subset_dp':
            from temporal_mst_exact import exact_mst_w
            result = asdict(exact_mst_w(graph, root, max_terminals=12, max_dp_cells=20_000_000))
        else:
            raise ValueError('unknown extended kernel')
        elapsed = time.perf_counter() - started
    phases['other_kernel_work'] = max(0.0, elapsed - math.fsum(phases.values()))
    return result, elapsed, phases, calls


def worker(args):
    started = time.perf_counter()
    graph = TemporalGraph.from_json(args.input)
    input_time = time.perf_counter() - started
    phase_started = time.perf_counter()
    required = temporal_reachable(graph, args.root, graph.edges)
    reachability_time = time.perf_counter() - phase_started
    settings = json.loads(args.settings)
    try:
        result, algorithm_time, phases, calls = measured_kernel(graph, args.root, args.method, settings)
        selected_ids = set(result.pop('edge_ids'))
        result.pop('reachable', None)
        selected = [edge for edge in graph.edges if edge.eid in selected_ids]
        phase_started = time.perf_counter()
        valid = validate_temporal_tree(graph, args.root, selected, required)
        covered = temporal_reachable(graph, args.root, selected)
        validation_time = time.perf_counter() - phase_started
        if not valid and args.method != 'snapshot_mst':
            raise RuntimeError('extended kernel returned an invalid physical temporal tree')
        result.update(validated=valid, status='feasible' if valid else 'model-mismatch', input_time_s=input_time, reachability_time_s=reachability_time, algorithm_time_s=algorithm_time, validation_time_s=validation_time, kernel_phases_s=phases, kernel_calls=calls, selected_events=len(selected), temporal_coverage_fraction=len(covered) / len(required), solution_sha256=hashlib.sha256('\n'.join(sorted(selected_ids)).encode()).hexdigest())
    except ValueError as error:
        if args.method != 'subset_dp' or not any(term in str(error) for term in ('limit', 'cells')):
            raise
        result = {'validated': False, 'status': 'size-guard', 'error': str(error), 'input_time_s': input_time, 'reachability_time_s': reachability_time}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(root=args.root, method=args.case_name or args.method, algorithm_method=args.method, settings=settings, vertices=len(graph.vertices), edges=len(graph.edges), reachable_count=len(required), wall_time_s=time.perf_counter() - started, peak_rss_mib=peak / (1024 * 1024 if sys.platform == 'darwin' else 1024))
    return result


def main():
    parser = argparse.ArgumentParser(description='Frozen-kernel baseline, phase, and sensitivity experiments')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--input')
    parser.add_argument('--root')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--case-name')
    parser.add_argument('--settings', default='{}')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    configuration = json.loads(args.config.read_text())
    sources = [Path(__file__), *[Path(name) for name in ('temporal_mst_prototype.py', 'temporal_mst_shared.py', 'temporal_mst_star.py', 'temporal_mst_cascade_star.py', 'temporal_mst_exact.py')]]
    metadata = {'date': time.strftime('%Y-%m-%d'), 'config': configuration, 'sha256': {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources}, 'input_sha256': {entry['input']: hashlib.sha256(Path(entry['input']).read_bytes()).hexdigest() for entry in configuration['datasets']}, 'python': sys.version, 'platform': platform.platform(), 'fresh_process': True, 'threads': 1, 'timing_scope': 'input + reachability + frozen kernel + original graph validation; imports excluded', 'profile_scope': 'nonoverlapping wrapper phases with unwrapped kernel work as residual', 'baseline_scope': 'EA branching optimizes arrival first; snapshot MST is a model-mismatch diagnostic; subset DP is a singleton-label exact reference, not author code'}
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    cases = configuration.get('cases', [{'name': method, 'algorithm': method} for method in configuration['methods']])
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for entry in configuration['datasets']:
        for root in entry['roots']:
            for case in cases:
                for repeat in range(configuration.get('repeats', 1)):
                    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--input', entry['input'], '--root', str(root), '--method', case['algorithm'], '--case-name', case['name'], '--settings', json.dumps(case)]
                    timeout = configuration.get('hard_timeout_s', 180)
                    try:
                        process = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=timeout)
                        record = json.loads(process.stdout) if process.returncode == 0 else {'root': root, 'method': case['name'], 'validated': False, 'status': 'worker-error', 'error': process.stderr[-2000:], 'returncode': process.returncode}
                    except subprocess.TimeoutExpired:
                        record = {'root': root, 'method': case['name'], 'validated': False, 'status': 'hard-timeout', 'wall_time_s': timeout}
                    record.update(dataset=entry['name'], repeat=repeat, scale_events=entry.get('scale_events'))
                    records.append(record)
                    args.output.write_text(json.dumps({'metadata': metadata, 'records': records}, indent=2, allow_nan=False) + '\n')
                    print(entry['name'], root, case['name'], record['status'], record.get('cost'), record.get('wall_time_s'), flush=True)
    print(f'Completed {len(records)} extended records', flush=True)


if __name__ == '__main__':
    main()
