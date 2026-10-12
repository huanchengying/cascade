from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from verify_artifact_20261008 import RESULT_FILES


DIRECTORY = Path('work/research_2026_10_08')


def load_records(name):
    return json.loads((DIRECTORY / name).read_text())['records']


def outcome(left, right):
    if math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
        return 'tie'
    return 'win' if left < right else 'loss'


def paired_summary(records, reference, treatment):
    pairs = defaultdict(dict)
    for record in records:
        pairs[record['dataset'], str(record['root'])][record['method']] = record
    selected = [(methods[reference], methods[treatment]) for methods in pairs.values()]
    results = [outcome(treated['cost'], baseline['cost']) for baseline, treated in selected]
    return {
        'queries': len(selected),
        'wins': results.count('win'),
        'ties': results.count('tie'),
        'losses': results.count('loss'),
        'mean_saving_pct': statistics.fmean(100 * (baseline['cost'] - treated['cost']) / baseline['cost'] for baseline, treated in selected),
        'median_saving_pct': statistics.median(100 * (baseline['cost'] - treated['cost']) / baseline['cost'] for baseline, treated in selected),
        'median_total_time_ratio': statistics.median(treated['wall_time_s'] / baseline['wall_time_s'] for baseline, treated in selected),
        'median_extra_time_s': statistics.median(treated['wall_time_s'] - baseline['wall_time_s'] for baseline, treated in selected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    optima = json.loads((DIRECTORY / 'exact_samples/optima.json').read_text())
    optimum = {record['name']: record['cost'] for record in optima}
    assert len(optimum) == 24
    assert all(math.isclose(record['subset_dp']['cost'], record['ikuta']['cost'], rel_tol=1e-8, abs_tol=1e-8) for record in optima)
    grouped = defaultdict(list)
    for record in load_records('submission_exact_quality_20261008.json') + load_records('submission_speedsteiner_exact_20261008.json'):
        if record.get('validated'):
            grouped[record['method'], record['dataset']].append(record['cost'])
    methods = defaultdict(list)
    for (method, dataset), costs in grouped.items():
        assert all(math.isclose(cost, costs[0], rel_tol=1e-8, abs_tol=1e-8) for cost in costs)
        methods[method].append(statistics.median(costs) / optimum[dataset])
    exact = {
        method: {
            'queries': len(ratios),
            'mean_cost_to_optimum': statistics.fmean(ratios),
            'optimum_hits': sum(math.isclose(ratio, 1, rel_tol=1e-8, abs_tol=1e-8) for ratio in ratios),
        }
        for method, ratios in methods.items()
    }
    assert [exact[method]['optimum_hits'] for method in ('cascade', 'star', 'speedsteiner', 'path_union')] == [19, 14, 10, 3]
    assert [f"{exact[method]['mean_cost_to_optimum']:.5f}" for method in ('cascade', 'star', 'speedsteiner', 'path_union')] == ['1.01237', '1.02205', '1.03339', '1.09765']

    converged_records = load_records('cascade_convergence_results.json')
    convergence = paired_summary(converged_records, 'star', 'cascade')
    star_records = [record for record in converged_records if record['method'] == 'star']
    assert all(record['rounds'] == record['accepted_moves'] + 1 for record in star_records)
    assert len(star_records) == 20
    star_costs = {(record['dataset'], str(record['root'])): record['cost'] for record in star_records}
    assert all(math.isclose(record['conservative_star_cost'], star_costs[record['dataset'], str(record['root'])], rel_tol=1e-8, abs_tol=1e-8) for record in converged_records if record['method'] == 'cascade')
    deadline_wins = sum(record['relaxed_deadline_vertices'] > 0 for record in converged_records if record['method'] == 'cascade' and outcome(record['cost'], star_costs[record['dataset'], str(record['root'])]) == 'win')
    assert (convergence['wins'], convergence['ties'], convergence['losses'], deadline_wins) == (12, 8, 0, 12)
    assert f"{convergence['median_saving_pct']:.4f}" == '0.0533'
    assert f"{convergence['median_total_time_ratio']:.3f}" == '1.098'
    convergence['verified_converged_baselines'] = len(star_records)
    convergence['wins_releasing_deadlines'] = deadline_wins

    budget_records = load_records('budget_curve_results.json')
    budget_summaries = {}
    for budget in (10, 30, 60):
        selected = [record for record in budget_records if record['budget_s'] == budget]
        for reference in ('cascade_warm', 'cascade_closure_weight'):
            budget_summaries[f'{budget}s:root_priced_vs_{reference}'] = paired_summary(selected, reference, 'cascade_closure_root')
    thirty = budget_summaries['30s:root_priced_vs_cascade_warm']
    assert (thirty['queries'], thirty['wins'], thirty['ties'], thirty['losses']) == (20, 20, 0, 0)
    assert f"{thirty['mean_saving_pct']:.3f}" == '8.644'
    assert budget_summaries['10s:root_priced_vs_cascade_closure_weight']['mean_saving_pct'] < 0

    structural = load_records('snap_structural_results.json')
    assert len(structural) == 16
    assert all(record['validated'] and math.isclose(record['cost'], record['reachable_count'] - 1) for record in structural)
    profiles = load_records('microprofile_results.json')
    small_citi = next(record for record in profiles if record['dataset'] == 'citibike-duration-1000' and record['method'] == 'cascade')
    assert (small_citi['reachable_count'], small_citi['selected_events']) == (3, 2)
    duration_profiles = {record['dataset']: record for record in profiles if record['method'] == 'cascade' and record['dataset'] in ('citibike-duration-100000', 'tlc-duration-100000')}
    assert all(record['kernel_calls']['fixed_parent_dp'] == 256 for record in duration_profiles.values())

    manuscript = Path('temporal_mst_submission_20261008.tex').read_text()
    title = re.search(r'\\title\{([^}]+)\}', manuscript).group(1)
    registration = Path('SUBMISSION_20261008.md').read_text()
    assert registration.split('## Working title\n\n', 1)[1].split('\n\n', 1)[0] == title
    abstract = manuscript.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0]
    plain_abstract = abstract.replace(r'\cascade{}', 'Cascade').replace(r'\cascade', 'Cascade').replace(r'$O(n+d_h\log n)$', 'O(n + d_h log n)').replace('$n$', 'n').replace('$d_h$', 'd_h').replace(r'\%', '%')
    registration_abstract = registration.split('## Abstract for registration\n\n', 1)[1].split('\n\nThis registration text', 1)[0]
    assert ' '.join(plain_abstract.split()) == ' '.join(registration_abstract.split())
    chapters = {path.stem: path.read_text() for path in sorted(Path('paper_20261008').glob('*.tex')) if path.stem != 'references'}
    headings = []
    for chapter, text in chapters.items():
        for match in re.finditer(r'\\textbf\{([^}]+)\}', text):
            if re.fullmatch(r'C[1-4]', match.group(1)):
                continue
            assert text[:match.start()].rstrip().endswith(r'\noindent'), (chapter, match.group(1))
            headings.append({'chapter': chapter, 'heading': match.group(1)})
    assert len(re.findall(r'\\subsection\{', chapters['optimizations'])) == 2
    assert r'\section{Conclusion}' in chapters['conclusion']
    assert 'C4: cost reductions and tradeoffs.' in chapters['introduction']
    assert 'twenty roots' not in chapters['introduction']
    assert 'S^+(h,q)' in chapters['cascade'] and 'U(h,q)' not in chapters['cascade']
    assert not any('PrefixPrice' in text for text in [manuscript, *chapters.values()])
    assert '19 optima' in manuscript and '8.64\\%' in manuscript
    assert 'Apple M4 Pro' in chapters['evaluation'] and 'Xeon' not in chapters['evaluation']
    total_records = sum(len(load_records(name)) for name in RESULT_FILES)
    assert total_records == 1079

    report = {
        'review_date': '2026-10-12',
        'status': 'passed',
        'title': title,
        'registration_title_and_abstract_match': True,
        'measurement_records': total_records,
        'exact_oracles_agree': 24,
        'exact_quality': exact,
        'verified_convergence': convergence,
        'budget_comparisons': budget_summaries,
        'snap_unit_weight_valid_optima': len(structural),
        'microprofile_citi_1k': {'reachable_vertices': small_citi['reachable_count'], 'tree_events': small_citi['selected_events']},
        'duration_parent_dp_calls': {name: record['kernel_calls']['fixed_parent_dp'] for name, record in duration_profiles.items()},
        'flush_left_bold_headings': headings,
        'chapter_four_subsections': 2,
        'final_section': 'Conclusion',
        'optimistic_saving_notation': 'S^+(h,q)',
        'upper_bound_notation': 'U',
        'measurement_platform': 'Apple M4 Pro; requested Xeon rerun remains pending',
        'result_sha256': {name: hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() for name in RESULT_FILES},
        'manuscript_source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path('temporal_mst_submission_20261008.tex'), *sorted(Path('paper_20261008').glob('*.tex'))]},
    }
    report_path = DIRECTORY / 'manuscript_consistency_20261009.json'
    if args.check:
        assert json.loads(report_path.read_text()) == report
    else:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'title', 'measurement_records', 'exact_oracles_agree', 'chapter_four_subsections')}, indent=2))


if __name__ == '__main__':
    main()
