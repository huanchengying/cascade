from __future__ import annotations

import difflib
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


CATALOG = [
    ('huang2015', 'Minimum Spanning Trees in Temporal Graphs', '10.1145/2723372.2723717', 'temporal spanning'),
    ('ikuta2016', 'Integer Programming Approach for Directed Minimum Spanning Tree Problem on Temporal Graphs', '10.1145/2980523.2980528', 'temporal spanning'),
    ('speedsteiner2025', 'SpeedSteiner: A Fast Approximation Algorithm for Directed Steiner Tree', '10.1145/3746252.3761415', 'shared costs'),
    ('bubboloni2025', 'On Computing Optimal Temporal Branchings and Spanning Subgraphs', '10.1016/j.jcss.2024.103596', 'temporal spanning'),
    ('yu2026', 'On Querying Minimum Spanning Tree in Temporal Graphs', '10.1007/s00778-026-00989-1', 'temporal spanning'),
    ('ge2023', 'An Efficient Dynamic Programming Algorithm for Finding Group Steiner Trees in Temporal Graphs', '10.1155/2023/1974161', 'shared costs'),
    ('husek2020', 'Approximation Algorithms for Steiner Tree Based on Star Contractions: A Unified View', '10.4230/LIPIcs.IPEC.2020.16', 'shared costs'),
    ('uchoa2010', 'Fast Local Search for Steiner Trees in Graphs', '10.1137/1.9781611972900.1', 'shared costs'),
    ('fisher1981', 'The Lagrangian Relaxation Method for Solving Integer Programming Problems', '10.1287/mnsc.27.1.1', 'optimization'),
    ('chronos2014', 'Chronos: a graph engine for temporal graph analysis', '10.1145/2592798.2592799', 'temporal systems'),
    ('raphtory2024', 'Raphtory: The temporal graph engine for Rust and Python', '10.21105/joss.05940', 'temporal systems'),
    ('tgx2024', 'Temporal Graph Analysis with TGX', '10.1145/3616855.3635694', 'temporal systems'),
    ('tea2023', 'TEA: A General-Purpose Temporal Graph Random Walk Engine', '10.1145/3552326.3567491', 'temporal systems'),
    ('memory2025', 'Temporal graph processing in modern memory hierarchies', '10.1016/j.is.2024.102462', 'temporal systems'),
    ('graphvault2024', 'GraphVault: A Temporal Graph Persistence Engine', '10.5220/0012556500003690', 'temporal systems'),
    ('temporalnetworks2012', 'Temporal networks', '10.1016/j.physrep.2012.03.001', 'temporal foundations'),
    ('holme2015', 'Modern temporal network theory: a colloquium', '10.1140/epjb/e2015-60657-4', 'temporal foundations'),
    ('streamgraphs2018', 'Stream graphs and link streams for the modeling of interactions over time', '10.1007/s13278-018-0537-7', 'temporal foundations'),
    ('charikar1999', 'Approximation algorithms for directed Steiner problems', None, 'shared costs'),
    ('siebert2020', 'A Simulated Annealing Algorithm for the Directed Steiner Tree Problem', '10.48550/arXiv.2002.03055', 'shared costs'),
    ('casteigts2012', 'Time-varying graphs and dynamic networks', '10.1080/17445760.2012.668546', 'temporal foundations'),
    ('kostakos2009', 'Temporal graphs', '10.1016/j.physa.2008.11.021', 'temporal foundations'),
    ('tang2010', 'Temporal distance metrics for social networks', None, 'temporal foundations'),
    ('pan2011', 'Path lengths, correlations, and centrality in temporal networks', None, 'temporal foundations'),
    ('accessibility2013', 'Unfolding accessibility provides a macroscopic approach to temporal networks', None, 'temporal foundations'),
    ('epidemic2012', 'Temporal networks and the dynamics of contagion', None, 'temporal applications'),
    ('epidemic2016', 'Temporal-network approach to epidemiology', None, 'temporal applications'),
    ('wu2014', 'Path problems in temporal graphs', None, 'temporal queries'),
    ('wu2016', 'Efficient algorithms for temporal path computation', None, 'temporal queries'),
    ('topchain2017', 'TopChain: Efficient Reachability Querying on Temporal Graphs', None, 'temporal queries'),
    ('traversals2016', 'Breadth-first search and depth-first search on temporal graphs', None, 'temporal queries'),
    ('core2015', 'Core decomposition of temporal graphs', None, 'temporal queries'),
    ('kovanen2011', 'Temporal motifs in time-dependent networks', None, 'temporal analysis'),
    ('motifs2017', 'Motifs in temporal networks', None, 'temporal analysis'),
    ('tgraph2017', 'TGraph: A temporal graph data management system', None, 'temporal systems'),
    ('algebra2016', 'A temporal graph algebra', None, 'temporal systems'),
    ('tegra2020', 'TEGRA: Efficient Ad-Hoc Analytics on Evolving Graphs', None, 'temporal systems'),
    ('kineograph2012', 'Kineograph: taking the pulse of a fast-changing and connected world', None, 'temporal systems'),
    ('graphone2019', 'GraphOne: A Data Store for Real-time Analytics on Evolving Graphs', None, 'graph systems'),
    ('graphbolt2019', 'GraphBolt: Dependency-Driven Synchronous Processing of Streaming Graphs', None, 'graph systems'),
    ('risgraph2021', 'RisGraph: A Real-Time Streaming System for Evolving Graphs to Support Sub-Millisecond Per-Update Analysis at Millions Ops/s', None, 'graph systems'),
    ('livegraph2020', 'LiveGraph: A Transactional Graph Storage System with Purely Sequential Adjacency List Scans', None, 'graph systems'),
    ('teseo2020', 'Teseo: A Graph Storage Engine for Mixed Transactional and Analytical Workloads', None, 'graph systems'),
    ('aspen2019', 'Aspen: A framework for efficient and scalable functional graph processing', None, 'graph systems'),
    ('stinger2012', 'STINGER: High performance data structure for streaming graphs', None, 'graph systems'),
    ('graphscope2021', 'GraphScope: A One-Stop Large-Scale Graph Computing System', None, 'graph systems'),
    ('graphscopeflex2024', 'GraphScope Flex: LEGO-like Graph Computing Stack', None, 'graph systems'),
    ('kickstarter2017', 'Kickstarter: Fast and Accurate Computations on Streaming Graphs via Trimmed Approximations', None, 'graph systems'),
    ('ligra2013', 'Ligra: a lightweight graph processing framework for shared memory', '10.1145/2442516.2442530', 'graph systems'),
    ('pregel2010', 'Pregel: a system for large-scale graph processing', None, 'graph systems'),
    ('graphframes2016', 'GraphFrames: An Integrated API for Mixing Graph and Relational Queries', None, 'graph systems'),
    ('graphalytics2016', 'Graphalytics: A Big Data Benchmark for Graph-Processing Platforms', None, 'evaluation'),
    ('dysat2020', 'DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks', None, 'temporal analysis'),
    ('dijkstra1959', 'A note on two problems in connexion with graphs', '10.1007/BF01386390', 'optimization'),
    ('kruskal1956', 'On the shortest spanning subtree of a graph and the traveling salesman problem', None, 'optimization'),
    ('prim1957', 'Shortest connection networks and some generalizations', None, 'optimization'),
    ('edmonds1967', 'Optimum branchings', '10.6028/jres.071B.032', 'optimization'),
    ('dreyfus1971', 'The Steiner problem in graphs', '10.1002/net.3230010302', 'shared costs'),
    ('zelikovsky1993', 'An 11/6-approximation algorithm for the network Steiner problem', None, 'shared costs'),
    ('duin1997', 'Efficient path and vertex exchange in Steiner tree algorithms', '10.1002/(SICI)1097-0037(199703)29:2<89::AID-NET3>3.0.CO;2-7', 'shared costs'),
    ('goemans1995', 'A general approximation technique for constrained forest problems', None, 'shared costs'),
    ('edmondskarp1972', 'Theoretical improvements in algorithmic efficiency for network flow problems', None, 'optimization'),
    ('highs2018', 'Parallelizing the dual revised simplex method', None, 'optimization'),
    ('scipy2020', 'SciPy 1.0: fundamental algorithms for scientific computing in Python', '10.1038/s41592-019-0686-2', 'evaluation'),
    ('miplib2021', 'MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Test Library', None, 'evaluation'),
]


def normalized(value):
    return re.sub(r'[^a-z0-9]+', ' ', html.unescape(re.sub('<[^>]*>', '', value)).lower()).strip()


def title_of(entry):
    return ': '.join(' '.join(entry.get(field, [])) for field in ('title', 'subtitle') if entry.get(field))


def lookup(item):
    key, title, doi, category = item
    provider = 'datacite' if doi and doi.lower().startswith(('10.4230/', '10.48550/')) else 'crossref'
    if provider == 'datacite':
        url = 'https://api.datacite.org/dois/' + urllib.parse.quote(doi, safe='/')
    elif doi:
        url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi, safe='/')
    else:
        url = 'https://api.crossref.org/works?' + urllib.parse.urlencode({'query.bibliographic': title, 'rows': 5})
    target = Path('work/literature') / f'{provider}_{key}_20261008.json'
    for attempt in range(3):
        try:
            raw = target.read_bytes() if target.is_file() else None
            if raw is not None and doi and provider == 'crossref':
                cached = json.loads(raw).get('message', {})
                if cached.get('DOI', '').lower() != doi.lower():
                    raw = None
            if raw is None:
                request = urllib.request.Request(url, headers={'User-Agent': 'TemporalMST-Research/1.0'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw = response.read()
            payload = json.loads(raw)
            target.write_bytes(raw)
            if provider == 'datacite':
                attributes = payload['data']['attributes']
                entries = [{'DOI': attributes['doi'], 'title': [attributes['titles'][0]['title']], 'author': [{'given': creator.get('givenName', ''), 'family': creator.get('familyName', creator['name'])} for creator in attributes['creators']], 'published': {'date-parts': [[attributes['publicationYear']]]}, 'container-title': [(attributes.get('container') or {}).get('title', attributes['publisher'])], 'type': 'posted-content' if 'arxiv' in attributes['doi'].lower() else 'proceedings-article', 'URL': 'https://doi.org/' + attributes['doi']}]
            else:
                entries = [payload['message']] if doi else payload['message']['items']
            candidate = max(entries, key=lambda entry: difflib.SequenceMatcher(None, normalized(title), normalized(title_of(entry))).ratio())
            score = difflib.SequenceMatcher(None, normalized(title), normalized(title_of(candidate))).ratio()
            accepted = bool(doi) or score >= 0.78
            return {'key': key, 'query_title': title, 'category': category, 'source_url': url, 'source': target.as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(), 'title_similarity': score, 'accepted': accepted, 'entry': candidate, 'candidate_titles': [title_of(entry) for entry in entries]}
        except Exception as error:
            if attempt == 2:
                return {'key': key, 'query_title': title, 'category': category, 'source_url': url, 'accepted': False, 'error': str(error)}
            time.sleep(attempt + 1)


def main():
    Path('work/literature').mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        records = list(executor.map(lookup, CATALOG))
    seen = set()
    for record in records:
        if record['accepted']:
            doi = record['entry']['DOI'].lower()
            if doi in seen:
                record['accepted'] = False
                record['duplicate_doi'] = True
            seen.add(doi)
        print(record['key'], record['accepted'], title_of(record['entry']) if 'entry' in record else record.get('error'), flush=True)
    output = Path('work/literature/verified_catalog_20261008.json')
    output.write_text(json.dumps({'date': '2026-10-08', 'records': records}, indent=2, ensure_ascii=False) + '\n')
    print(f"Accepted {sum(record['accepted'] for record in records)} distinct verified DOI records", flush=True)


if __name__ == '__main__':
    main()
