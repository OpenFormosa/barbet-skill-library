"""Recompute saved v26 score minima and query/family folds without inference."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sqlite3

from scripts import audit_barbet_continuation_v1 as audit
from scripts import publish_barbet_only_v1 as release

DBS = [
    'memory-harness-v2-v26-complete-selected-v1/selection.sqlite',
    'memory-harness-v2-natural-instructions-v1/selected-v1/selection.sqlite',
    'memory-harness-v2-opencode-v1/selected-v1/selection.sqlite',
    'memory-harness-v2-ifm-v1/math-full-v1/selected-v1/selection.sqlite',
    'memory-harness-v2-ifm-v1/code-full-v1/selected-v1/selection.sqlite',
]


def scores(relative):
    path = release.base.WORK / 'reports' / relative
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    decisions = {r[0]: (r[1], r[2], json.loads(r[3])) for r in db.execute('SELECT * FROM decisions')}
    counts = Counter(); completed = set(); current = None; values = []
    def finish(group, rows):
        if group is None: return
        status, winner, report = decisions[group]
        if status != 'selected': counts['held_groups_with_partial_scores'] += 1; return
        if len(rows) != report['candidates']:
            raise ValueError('Incomplete saved competing scores')
        if len({x['protocol_sha256'] for x in rows}) != 1:
            raise ValueError('Mixed numerical score protocol')
        best = min(rows, key=lambda x: (x['mean_nll'], x['record_id']))
        if best['candidate_identity'] != winner or abs(best['mean_nll'] - report['mean_nll']) > 1e-12:
            raise ValueError('Saved selected answer is not the minimum')
        completed.add(group); counts['selected_groups_recomputed'] += 1
    for identity, group, rid, raw in db.execute('SELECT * FROM scores ORDER BY input_hash'):
        value = json.loads(raw)
        if group != current:
            finish(current, values); current = group; values = []
        if (value['status'] != 'scored' or value['input_hash'] != group or value['record_id'] != rid
                or value['candidate_identity'] != identity or value['answer_tokens'] <= 0
                or not math.isfinite(value['mean_nll']) or value['mean_nll'] < 0
                or not math.isfinite(value['ppl']) or value['ppl'] < 1
                or abs(math.log(value['ppl']) - value['mean_nll']) > 1e-10):
            raise ValueError('Invalid saved native v26 score')
        values.append(value); counts['scores_checked'] += 1
    finish(current, values)
    expected = {key for key, value in decisions.items() if value[0] == 'selected'}
    if completed != expected: raise ValueError('Missing selected score groups')
    db.close()
    return {'database': relative, 'sha256': release.base.file_sha(path), 'counts': dict(counts)}


def folds():
    path = release.parent.STATE / 'public-index.sqlite'
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    db.execute('PRAGMA temp_store=MEMORY')
    result = {'counts': {}, 'cross_split_query_or_family': {}}
    for key in ['record_id', 'query_hash', 'family_id']:
        result['counts'][key] = db.execute(f'SELECT COUNT(DISTINCT {key}) FROM records').fetchone()[0]
    for key in ['query_hash', 'family_id']:
        found = db.execute(f'SELECT {key} FROM records GROUP BY {key} HAVING MIN(split)!=MAX(split) LIMIT 1').fetchone()
        if found: raise ValueError('Cross-split leakage: ' + key)
        result['cross_split_query_or_family'][key] = 0
    query_holds = set(); provenance = []
    for path in (release.parent.DEST / 'reports').rglob('*hold*.json'):
        content = release.read(path)
        if isinstance(content, dict) and 'query_hashes' in content:
            hashes = content['query_hashes']
            if not all(isinstance(x, str) and len(x) == 64 for x in hashes):
                raise ValueError('Unexpected public query-wide hold shape')
            query_holds.update(hashes)
            provenance.append({'path': str(path.relative_to(release.parent.DEST)),
                               'sha256': release.base.file_sha(path), 'queries': len(hashes)})
    if query_holds and db.execute('SELECT 1 FROM records WHERE query_hash IN (' + ','.join('?' for _ in query_holds) + ') LIMIT 1', sorted(query_holds)).fetchone():
        raise ValueError('Historical query-wide hold reintroduced')
    result.update(query_wide_holds_checked=len(query_holds), hold_files=provenance,
                  source_counts=dict(db.execute('SELECT source,COUNT(*) FROM records GROUP BY source')))
    db.close(); return result


if __name__ == '__main__':
    path = audit.STATE / 'saved-selection-and-folds.json'
    if path.exists(): raise ValueError('Fresh saved-selection audit already completed')
    release.base.save(audit.STATE / 'saved-selection-policy.json', {
        'driver_sha256': release.base.file_sha(Path(__file__)), 'score_databases': DBS,
        'model_requests': 0, 'all_original_scores_and_first_holds_unchanged': True})
    with ThreadPoolExecutor(max_workers=5) as pool:
        checked = list(pool.map(scores, DBS))
    for item in checked: print(json.dumps(item), flush=True)
    result = {'saved_score_checks': checked, 'folds': folds(), 'model_requests': 0,
              'scores_recomputed_from_model': False, 'saved_minima_recomputed': True}
    release.base.save(path, result)
    print(json.dumps({'all_saved_minima_and_folds_passed': True,
                      'selected_groups_checked': sum(x['counts']['selected_groups_recomputed'] for x in checked)}), flush=True)
