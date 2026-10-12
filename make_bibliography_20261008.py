from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path


DIRECTORY = Path('work/literature')


def plain(value):
    return ' '.join(html.unescape(re.sub('<[^>]*>', '', value)).split()).strip().rstrip(':')


def escape(value):
    replacements = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(character, character) for character in plain(str(value)))


def main():
    catalog = json.loads((DIRECTORY / 'verified_catalog_20261008.json').read_text())
    selected = [record for record in catalog['records'] if record['accepted']]
    seen = set()
    lines = [r'\begin{thebibliography}{99}']
    provenance = []
    correction_path = DIRECTORY / 'bibliography_corrections_20261010.json'
    corrections = {record['key']: record for record in json.loads(correction_path.read_text())['records']} if correction_path.is_file() else {}
    for record in selected:
        entry = record['entry']
        doi = entry['DOI'].lower()
        if doi in seen or not entry.get('author'):
            raise ValueError('duplicate DOI or absent author metadata')
        seen.add(doi)
        source = Path(record['source'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('citation metadata cache changed')
        title = ': '.join(plain(' '.join(entry[field])) for field in ('title', 'subtitle') if entry.get(field))
        if record['key'] == 'graphone2019':
            title = re.sub(r'G\s*raph\s*O\s*ne', 'GraphOne', title)
        authors = [' '.join(filter(None, (author.get('given', ''), author.get('family', author.get('name', ''))))) for author in entry['author']]
        correction = corrections.get(record['key'])
        if correction:
            if authors != correction['canonical_authors'] or hashlib.sha256(Path(correction['source']).read_bytes()).hexdigest() != correction['source_sha256']:
                raise ValueError('author correction evidence differs from cached metadata')
            authors = correction['corrected_authors']
        if any(not author for author in authors):
            raise ValueError('empty author name')
        displayed = authors if len(authors) <= 10 else authors[:8] + ['et al.']
        author_text = ', '.join(escape(author) for author in displayed)
        year = next(entry[field]['date-parts'][0][0] for field in ('published-print', 'published', 'issued') if entry.get(field, {}).get('date-parts'))
        venue = plain(' '.join(entry.get('container-title', [])))
        publication = r'\emph{' + escape(venue) + '}'
        numbers = [entry.get('volume'), entry.get('issue')]
        if numbers[0]:
            publication += ' ' + escape(numbers[0])
            if numbers[1]:
                publication += ', ' + escape(numbers[1])
        pages = entry.get('page', entry.get('article-number'))
        if pages:
            publication += ', ' + escape(pages).replace('-', '--')
        formatted_title = escape(title)
        if record['key'] == 'speedsteiner2025':
            formatted_title = r'SpeedSteiner: A Fast $O(k^{1/2})$-Approximation Algorithm for Directed Steiner Tree'
        author_ending = '' if author_text.endswith('.') else '.'
        lines.extend([r'\bibitem{' + record['key'] + '} ' + author_text + author_ending + f' {year}.', formatted_title + '. ' + publication + '.', r'\url{https://doi.org/' + doi + '}'])
        provenance.append({'key': record['key'], 'title': title, 'year': year, 'authors': authors, 'doi': doi, 'venue': venue, 'source': source.as_posix(), 'source_url': record['source_url'], 'source_sha256': record['sha256'], 'verification': 'canonical DOI metadata and title inspected; displayed years follow publication metadata'})
        if correction:
            provenance[-1]['correction_sources'] = [
                {'source': correction_path.as_posix(), 'source_sha256': hashlib.sha256(correction_path.read_bytes()).hexdigest()},
                {'source': correction['source'], 'source_sha256': correction['source_sha256']},
            ]
    datasets = [
        ('snap', 'Jure Leskovec and Andrej Krevl', 2014, 'SNAP Datasets: Stanford Large Network Dataset Collection', 'https://snap.stanford.edu/data/'),
        ('snapalpha', 'Stanford Network Analysis Project', 'n.d', 'Bitcoin Alpha trust weighted temporal network', 'https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html'),
        ('snapotc', 'Stanford Network Analysis Project', 'n.d', 'Bitcoin OTC trust weighted temporal network', 'https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html'),
        ('snapcollege', 'Stanford Network Analysis Project', 'n.d', 'CollegeMsg temporal network', 'https://snap.stanford.edu/data/CollegeMsg.html'),
        ('snapemail', 'Stanford Network Analysis Project', 'n.d', 'email-Eu-core temporal network', 'https://snap.stanford.edu/data/email-Eu-core-temporal.html'),
        ('citibike', 'Citi Bike', 2026, 'Citi Bike System Data', 'https://citibikenyc.com/system-data'),
        ('tlc', 'New York City Taxi and Limousine Commission', 2025, 'TLC Trip Record Data', 'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page'),
    ]
    for key, author, year, title, url in datasets:
        lines.extend([r'\bibitem{' + key + '} ' + escape(author) + f'. {year}. ' + escape(title) + '.', r'\url{' + url + '}. Accessed October 8, 2026.'])
        record = {'key': key, 'title': title, 'year': year, 'authors': [author], 'source_url': url, 'verification': 'official data provider; source and processed-input provenance in the artifact'}
        cached_pages = {'snapalpha': 'alpha', 'snapotc': 'otc', 'snapcollege': 'college', 'snapemail': 'email'}
        if key in cached_pages:
            source = DIRECTORY / (cached_pages[key] + '_page_20261008.html')
            record.update(source=source.as_posix(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        provenance.append(record)
    lines.append(r'\end{thebibliography}')
    Path('paper_20261008').mkdir(exist_ok=True)
    Path('paper_20261008/references.tex').write_text('\n'.join(lines) + '\n')
    (DIRECTORY / 'citation_provenance_20261008.json').write_text(json.dumps({'retrieved': catalog['date'], 'updated': catalog.get('updated', catalog['date']), 'distinct_references': len(provenance), 'records': provenance}, indent=2, ensure_ascii=False) + '\n')
    print(f'Generated {len(provenance)} distinct references, including {len(seen)} verified DOIs')


if __name__ == '__main__':
    main()
