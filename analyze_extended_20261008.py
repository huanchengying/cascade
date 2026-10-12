from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_cascade_results import audit_hashes


DIRECTORY = Path('work/research_2026_10_08')
LABELS = {'citibike-duration': 'Citi duration', 'tlc-duration': 'TLC duration', 'tlc-amount': 'TLC amount'}
METHODS = {'path_union': 'Independent paths', 'ea_branching': 'EA tree', 'snapshot_mst': 'Static projection', 'star': 'Conservative star', 'cascade': 'Cascade', 'subset_dp': 'Subset DP'}
PHASES = ('input_time_s', 'reachability_time_s', 'path_construction', 'parent_exchange', 'conservative_star', 'candidate_scoring', 'fixed_parent_dp', 'apply_and_validate', 'other_kernel_work', 'validation_time_s')


def geometric_mean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values)) if values else None


def checked_load(path):
    payload = json.loads(path.read_text())
    checks = audit_hashes(payload, Path.cwd())
    if not checks or not all(check['matches'] for check in checks):
        raise ValueError(f'extended input or source hash mismatch: {path}')
    configuration = payload['metadata']['config']
    expected = {(entry['name'], str(root), method, repeat) for entry in configuration['datasets'] for root in entry['roots'] for method in configuration['methods'] for repeat in range(configuration.get('repeats', 1))}
    keys = [(record['dataset'], str(record['root']), record['method'], record['repeat']) for record in payload['records']]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('incomplete or duplicate extended matrix')
    return payload, checks


def profile_rows(payload):
    groups = defaultdict(dict)
    for record in payload['records']:
        groups[record['dataset'], str(record['root'])][record['method']] = record
    rows = []
    for (dataset, root), records in sorted(groups.items()):
        cascade = records['cascade']
        if not cascade.get('validated'):
            raise ValueError('missing feasible microprofile reference')
        for method, record in records.items():
            valid = bool(record.get('validated'))
            row = {'dataset': dataset, 'domain': dataset.rsplit('-', 1)[0], 'root': root, 'method': method, 'events': record['scale_events'], 'status': record['status'], 'validated': valid, 'cost': record.get('cost'), 'cost_to_cascade': record['cost'] / cascade['cost'] if valid and cascade['cost'] else None, 'wall_s': record.get('wall_time_s'), 'algorithm_s': record.get('algorithm_time_s'), 'rss_mib': record.get('peak_rss_mib'), 'temporal_coverage_fraction': record.get('temporal_coverage_fraction'), 'reachable': record.get('reachable_count'), 'selected_events': record.get('selected_events')}
            phases = record.get('kernel_phases_s', {})
            for phase in PHASES:
                row[phase] = record.get(phase, phases.get(phase, 0.0))
            if valid and not math.isclose(math.fsum(phases.values()), record['algorithm_time_s'], rel_tol=1e-7, abs_tol=1e-6):
                raise ValueError('phase durations overlap or fail to sum to the kernel')
            rows.append(row)
    return rows


def sensitivity_rows(payload):
    defaults = {(record['dataset'], str(record['root'])): record for record in payload['records'] if record['method'] == 'default'}
    rows = []
    for record in payload['records']:
        default = defaults[record['dataset'], str(record['root'])]
        valid = bool(record.get('validated'))
        rows.append({'dataset': record['dataset'], 'root': str(record['root']), 'case': record['method'], 'status': record['status'], 'validated': valid, 'cost': record.get('cost'), 'saving_vs_default_pct': 100 * (default['cost'] - record['cost']) / default['cost'] if valid else None, 'wall_s': record.get('wall_time_s'), 'rss_mib': record.get('peak_rss_mib'), 'accepted_moves': record.get('accepted_moves'), 'relaxed_deadlines': record.get('relaxed_deadline_vertices'), 'evaluated_parents': record.get('evaluated_parent_events')})
    return rows


def exact_rows(payload, optima):
    optimum = {entry['name']: entry['cost'] for entry in optima}
    grouped = defaultdict(list)
    for record in payload['records']:
        grouped[record['method']].append(record)
    rows = []
    for method, records in sorted(grouped.items()):
        feasible = [record for record in records if record.get('validated')]
        ratios = [record['cost'] / optimum[record['dataset']] for record in feasible]
        if any(ratio < 1 - 1e-8 for ratio in ratios):
            raise ValueError('additional method beats dual-oracle optimum')
        if method == 'subset_dp' and any(not math.isclose(ratio, 1, rel_tol=1e-8) for ratio in ratios):
            raise ValueError('subset DP contradicts the frozen optimum')
        rows.append({'method': method, 'units': len(records), 'available': len(feasible), 'mean_cost_ratio': statistics.fmean(ratios) if ratios else None, 'geometric_mean_cost_ratio': geometric_mean(ratios), 'worst_cost_ratio': max(ratios) if ratios else None, 'optimal_hits': sum(math.isclose(ratio, 1, rel_tol=1e-8) for ratio in ratios), 'median_wall_s': statistics.median(record['wall_time_s'] for record in feasible) if feasible else None, 'median_rss_mib': statistics.median(record['peak_rss_mib'] for record in feasible) if feasible else None})
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_figures(profile, sensitivity, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    colors = {'path_union': '#7f7f7f', 'ea_branching': '#b68f00', 'star': '#1f77b4', 'cascade': '#b2182b'}
    figure, axes = plt.subplots(1, 3, figsize=(7, 2.15))
    for domain, axis in zip(LABELS, axes):
        for method in colors:
            selected = sorted((row for row in profile if row['domain'] == domain and row['method'] == method and row['validated']), key=lambda row: row['events'])
            axis.plot([row['events'] for row in selected], [row['wall_s'] for row in selected], marker='o', markersize=3, label=METHODS[method], color=colors[method])
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_title(LABELS[domain])
        axis.set_xlabel('Sampled edges')
        axis.grid(alpha=.2)
    axes[0].set_ylabel('End-to-end time (s)')
    axes[1].legend(fontsize=6, loc='best')
    figure.tight_layout(pad=.5)
    for suffix in ('pdf', 'png'):
        figure.savefig(directory / f'micro_scale_20261008.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(figure)

    selected = [next(row for row in profile if row['domain'] == domain and row['events'] == 100000 and row['method'] == 'cascade') for domain in LABELS]
    figure, axes = plt.subplots(1, 2, figsize=(7, 2.1))
    phase_groups = [('Input + reachability', ('input_time_s', 'reachability_time_s')), ('Paths + exchange', ('path_construction', 'parent_exchange')), ('Conservative star', ('conservative_star',)), ('Candidate scoring', ('candidate_scoring',)), ('Fixed-parent DP', ('fixed_parent_dp',)), ('Other + validation', ('apply_and_validate', 'other_kernel_work', 'validation_time_s'))]
    left = np.zeros(len(selected))
    for label, phases in phase_groups:
        values = [100 * math.fsum(row[phase] for phase in phases) / row['wall_s'] for row in selected]
        axes[0].barh([LABELS[row['domain']] for row in selected], values, left=left, label=label)
        left += values
    axes[0].set_xlabel('Share of end-to-end time (%)')
    axes[0].set_xlim(0, 100)
    axes[0].legend(fontsize=5.5, ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.36))
    for domain in LABELS:
        values = sorted((row for row in profile if row['domain'] == domain and row['method'] == 'cascade'), key=lambda row: row['events'])
        axes[1].plot([row['events'] for row in values], [row['rss_mib'] for row in values], marker='o', markersize=3, label=LABELS[domain])
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Sampled edges')
    axes[1].set_ylabel('Peak process RSS (MiB)')
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=.2)
    figure.tight_layout(pad=.8)
    for suffix in ('pdf', 'png'):
        figure.savefig(directory / f'micro_phases_20261008.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7, 2.1))
    fields = [('max_parent_evaluations', [16, 32, 128, 512]), ('conservative_rounds', [0, 4, 16, 32])]
    for axis, (field, values) in zip(axes, fields):
        for dataset in LABELS:
            entries = {row['case']: row for row in sensitivity if row['dataset'] == dataset and row['validated']}
            selected = [entries['default'] if (field == 'max_parent_evaluations' and value == 128) or (field == 'conservative_rounds' and value == 4) else entries[f'{field}-{value}'] for value in values]
            axis.plot(values, [row['saving_vs_default_pct'] for row in selected], marker='o', markersize=3, label=LABELS[dataset])
        axis.set_xlabel('Parent evaluations per round' if field == 'max_parent_evaluations' else 'Conservative-star round cap')
        axis.set_ylabel('Cost saving vs default (%)')
        axis.grid(alpha=.2)
    axes[0].set_xscale('log', base=2)
    axes[1].legend(fontsize=6)
    figure.tight_layout(pad=.5)
    for suffix in ('pdf', 'png'):
        figure.savefig(directory / f'sensitivity_20261008.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(figure)


def write_tables(profile, sensitivity, exact, directory):
    lines = [r'\begin{table}[t]', r'\centering', r'\caption{Additional exact-sample baselines. EA prioritizes arrival; subset DP encodes singleton labels. Neither row is an author-code timing reproduction.}', r'\label{tab:additional-exact}', r'\begin{tabular}{lrrrr}', r'\toprule', r'Method & Avail. & Mean/OPT & Optimal & Time (s)\\', r'\midrule']
    for row in exact:
        lines.append(f"{METHODS[row['method']]} & {row['available']}/{row['units']} & {row['mean_cost_ratio']:.4f} & {row['optimal_hits']}/{row['available']} & {row['median_wall_s']:.4f}\\\\")
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (directory / 'additional_exact_table.tex').write_text('\n'.join(lines) + '\n')
    lines = [r'\begin{table}[t]', r'\centering', r'\caption{One root per 100K-edge domain. Cost ratios use default Cascade; end-to-end time is in seconds and peak process RSS in MiB. Static-projection results failing temporal validation have no reported cost ratio.}', r'\label{tab:micro}', r'\begin{tabular}{lrrrr}', r'\toprule', r'Domain / metric & Paths & EA & Star & Cascade\\', r'\midrule']
    for domain in LABELS:
        selected = [next(row for row in profile if row['domain'] == domain and row['method'] == method and row['events'] == 100000) for method in ('path_union', 'ea_branching', 'star', 'cascade')]
        label = {'citibike-duration': 'Citi', 'tlc-duration': 'TLC dur.', 'tlc-amount': 'TLC amt.'}[domain]
        for metric, field, precision in (('cost', 'cost_to_cascade', 3), ('time', 'wall_s', 3), ('RSS', 'rss_mib', 1)):
            values = ' & '.join(f"{row[field]:.{precision}f}" for row in selected)
            lines.append(f"{label} {metric} & {values}\\\\")
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (directory / 'micro_table.tex').write_text('\n'.join(lines) + '\n')
    lines = [r'\begin{table}[t]', r'\centering', r'\caption{Sensitivity: paired change from default, aggregated descriptively over three roots. Positive saving lowers cost. All other settings remain fixed.}', r'\label{tab:sensitivity}', r'\begin{tabular}{lrr}', r'\toprule', r'Changed setting & Mean saving & Median wall\\', r'\midrule']
    groups = defaultdict(list)
    for row in sensitivity:
        groups[row['case']].append(row)
    for name, rows in groups.items():
        label = name.replace('max_parent_evaluations-', 'P=').replace('max_hubs-', 'H=').replace('conservative_rounds-', 'Star=').replace('cascade_rounds-', 'Cascade=')
        lines.append(f"{label} & {statistics.fmean(row['saving_vs_default_pct'] for row in rows):.3f}\\% & {statistics.median(row['wall_s'] for row in rows):.3f} s\\\\")
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (directory / 'sensitivity_table.tex').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    loaded = [checked_load(args.directory / name) for name in ('microprofile_results.json', 'sensitivity_results.json', 'additional_exact_results.json')]
    profile = profile_rows(loaded[0][0])
    sensitivity = sensitivity_rows(loaded[1][0])
    exact = exact_rows(loaded[2][0], json.loads((args.directory / 'exact_samples/optima.json').read_text()))
    payload = {'hash_checks': [check for _, checks in loaded for check in checks], 'profile': profile, 'sensitivity': sensitivity, 'additional_exact': exact, 'total_records': sum(len(item['records']) for item, _ in loaded), 'statistics': 'one repeat per unit; paired descriptive results; no confidence intervals'}
    (args.directory / 'extended_summary_20261008.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    for prefix, rows in (('microprofile', profile), ('sensitivity', sensitivity), ('additional_exact', exact)):
        write_csv(args.directory / f'{prefix}_summary_20261008.csv', rows)
    write_figures(profile, sensitivity, args.directory)
    write_tables(profile, sensitivity, exact, args.directory)
    lines = ['# Extended evaluation', '', 'Frozen source/input hashes and all three complete matrices verified.', '', 'Snapshot MST results are feasibility diagnostics; no infeasible cost enters a quality comparison.', '', '| Exact baseline | Available | Mean/OPT | Optima | Median time |', '|---|---:|---:|---:|---:|']
    for row in exact:
        lines.append(f"| {row['method']} | {row['available']}/{row['units']} | {row['mean_cost_ratio']:.6f} | {row['optimal_hits']} | {row['median_wall_s']:.4f} s |")
    lines.extend(['', 'Full per-unit phase, space, and sensitivity data are in the accompanying JSON/CSV files.', ''])
    (args.directory / 'extended_summary_20261008.md').write_text('\n'.join(lines))
    print(f"Analyzed {payload['total_records']} extended records")


if __name__ == '__main__':
    main()
