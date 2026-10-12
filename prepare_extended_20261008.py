import hashlib
import json
from pathlib import Path


DIRECTORY = Path('work/research_2026_10_08')
SOURCES = [
    {'name': 'citibike-duration', 'input': str(DIRECTORY / 'data/citibike-duration-100k.json'), 'roots': ['6197.08']},
    {'name': 'tlc-duration', 'input': 'work/data/processed/nyc-tlc-yellow-2025-01-duration-100k.json', 'roots': ['161']},
    {'name': 'tlc-amount', 'input': 'work/data/processed/nyc-tlc-yellow-2025-01-total-amount-100k.json', 'roots': ['161']},
]


def main():
    directory = DIRECTORY / 'extended_inputs'
    directory.mkdir(parents=True, exist_ok=True)
    datasets = []
    for source in SOURCES:
        payload = json.loads(Path(source['input']).read_text())
        ranked = sorted(payload['edges'], key=lambda edge: (hashlib.blake2b(str(edge['id']).encode(), digest_size=8).digest(), str(edge['id'])))
        for count in (1000, 10000, 30000, 100000):
            if count == 100000:
                target = Path(source['input'])
            else:
                selected = sorted(ranked[:count], key=lambda edge: (edge['start'], str(edge['id'])))
                vertices = sorted(set(source['roots']) | {edge['u'] for edge in selected} | {edge['v'] for edge in selected})
                target = directory / f"{source['name']}-{count}.json"
                target.write_text(json.dumps({'vertices': vertices, 'edges': selected, 'metadata': {'source': source['input'], 'source_sha256': hashlib.sha256(Path(source['input']).read_bytes()).hexdigest(), 'sampling': 'nested smallest BLAKE2b-64 digests of physical event IDs from frozen 100K input', 'scale_events': count}}, separators=(',', ':'), allow_nan=False) + '\n')
            datasets.append({'name': f"{source['name']}-{count}", 'input': target.as_posix(), 'roots': source['roots'], 'scale_events': count})
    profile = {'datasets': datasets, 'methods': ['path_union', 'ea_branching', 'snapshot_mst', 'star', 'cascade'], 'repeats': 1, 'hard_timeout_s': 180}
    defaults = {'algorithm': 'cascade', 'conservative_rounds': 4, 'cascade_rounds': 2, 'max_hubs': 64, 'max_parent_evaluations': 128}
    cases = [{'name': 'default', **defaults}]
    for field, values in [('max_parent_evaluations', [16, 32, 512]), ('max_hubs', [8, 32, 128]), ('conservative_rounds', [0, 16, 32]), ('cascade_rounds', [0, 1, 4])]:
        cases.extend({'name': f'{field}-{value}', **defaults, field: value} for value in values)
    sensitivity = {'datasets': SOURCES, 'methods': [case['name'] for case in cases], 'cases': cases, 'repeats': 1, 'hard_timeout_s': 180}
    exact = json.loads((DIRECTORY / 'exact_samples/config.json').read_text())
    additional = {'datasets': exact['datasets'], 'methods': ['ea_branching', 'subset_dp'], 'repeats': 1, 'hard_timeout_s': 300}
    for name, configuration in [('microprofile_config', profile), ('sensitivity_config', sensitivity), ('additional_exact_config', additional)]:
        (DIRECTORY / f'{name}.json').write_text(json.dumps(configuration, indent=2) + '\n')
    print('Prepared 60 microprofile, 39 sensitivity, and 48 additional-exact records')


if __name__ == '__main__':
    main()
