"""Freeze all candidate-family aliases of the verified v26 release for expansion.

The component graph includes non-winning alternatives, so a later source cannot
cross a frozen fold through an alias that happened not to win the PPL ranking.
"""
from collections import Counter
import json
import sqlite3

from scripts import build_memory_sft_v2 as base
from scripts.build_v26_complete_release import STATE as PRIOR
from scripts.extend_v26_overlap_selection import STATE as PRIOR_SCORE
from scripts.build_memory_sft_wave2 import STATE as ANCESTOR
from scripts.build_natural_instructions_memory_v1 import STATE as NI

STATE = NI / 'frozen-prior-v1'


class Components:
    def __init__(self): self.parent = {}

    def find(self, value):
        parent = self.parent; parent.setdefault(value, value); root = value
        while parent[root] != root: root = parent[root]
        while parent[value] != value:
            old = parent[value]; parent[value] = root; value = old
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)


def add_frozen(ties, conflicts, root, family, split):
    if root in ties and ties[root][1] != split:
        conflicts.add(root)
    else:
        ties[root] = min(ties.get(root, (family, split)), (family, split))


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'frozen-aliases.sqlite').exists(): raise ValueError('fold freeze already started; inspect immutable artifacts')
    package_path = PRIOR / 'public-package-summary.json'; package = json.loads(package_path.read_text())
    if not package['all_published_query_folds_preserved']: raise ValueError('prior release not verified')
    graph = Components()
    db = sqlite3.connect(f'file:{PRIOR_SCORE / "candidates.sqlite"}?mode=ro', uri=True)
    for family, template, query, condition in db.execute('SELECT source_family_id,template_family_id,query_hash,input_hash FROM candidates'):
        node = 'family:' + family
        for other in ['family:' + template, 'query:' + query, 'input:' + condition]: graph.union(node, other)
    db.close(); ties = {}; conflicts = set(); ancestor_only = []
    old = sqlite3.connect(f'file:{ANCESTOR / "frozen-aliases.sqlite"}?mode=ro', uri=True)
    for node, family, split in old.execute('SELECT node,family_id,split FROM aliases'):
        if node in graph.parent: add_frozen(ties, conflicts, graph.find(node), family, split)
        else: ancestor_only.append((node, family, split))
    standalone_conflicts = []
    for node, in old.execute('SELECT node FROM conflicts'):
        if node in graph.parent: conflicts.add(graph.find(node))
        else: standalone_conflicts.append((node,))
    old.close()
    # Active release family IDs take precedence over old alias IDs, while every
    # previously assigned split must remain identical.
    current = {}; active = sqlite3.connect(f'file:{PRIOR / "public-index.sqlite"}?mode=ro', uri=True)
    for query, family, split in active.execute('SELECT query_hash,family_id,split FROM records'):
        root = graph.find('query:' + query)
        if root in conflicts or (root in ties and ties[root][1] != split):
            raise ValueError('active record conflicts with an ancestral fold')
        if root in current and current[root] != (family, split):
            raise ValueError('active component has inconsistent frozen family or fold')
        current[root] = (family, split)
    active.close(); ties.update(current)
    out = sqlite3.connect(STATE / 'frozen-aliases.sqlite')
    out.execute('CREATE TABLE aliases(node TEXT PRIMARY KEY,family_id TEXT,split TEXT)')
    out.execute('CREATE TABLE conflicts(node TEXT PRIMARY KEY)')
    out.executemany('INSERT INTO aliases VALUES (?,?,?)', ancestor_only)
    out.executemany('INSERT INTO conflicts VALUES (?)', standalone_conflicts)
    buffer = []; bad = []; counts = Counter()
    for node in graph.parent:
        root = graph.find(node)
        if root in ties:
            family, split = ties[root]; buffer.append((node, family, split))
        if root in conflicts: bad.append((node,))
        if len(buffer) + len(bad) >= 10000:
            out.executemany('INSERT INTO aliases VALUES (?,?,?)', buffer)
            out.executemany('INSERT INTO conflicts VALUES (?)', bad); out.commit(); buffer = []; bad = []
    out.executemany('INSERT INTO aliases VALUES (?,?,?)', buffer)
    out.executemany('INSERT INTO conflicts VALUES (?)', bad); out.commit()
    counts['frozen_aliases'] = out.execute('SELECT COUNT(*) FROM aliases').fetchone()[0]
    counts['conflicting_aliases'] = out.execute('SELECT COUNT(*) FROM conflicts').fetchone()[0]
    counts['active_components'] = len(current); counts['all_candidate_component_nodes'] = len(graph.parent)
    out.close()
    base.save(STATE / 'summary.json', {'counts': dict(counts), 'prior_package_sha256': base.file_sha(package_path),
        'prior_candidate_summary_sha256': base.file_sha(PRIOR_SCORE / 'candidate-index-summary.json'),
        'frozen_database_sha256': base.file_sha(STATE / 'frozen-aliases.sqlite'),
        'all_candidate_aliases_included': True, 'all_ancestral_folds_preserved': True,
        'source': str(PRIOR), 'new_training_rows': 0})
    print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__': main()
