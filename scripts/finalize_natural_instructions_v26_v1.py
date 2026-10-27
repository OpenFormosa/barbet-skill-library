"""Select one native-v26 winner per NI input with frozen prior folds.

Example/target overlap is a separate quality hold, not a fabricated score.
The generic finalizer independently recomputes every saved minimum and refuses
partial competing groups, different literal conditions, or frozen-fold bridges.
"""
import argparse
import json
from pathlib import Path
import sqlite3

from scripts import build_memory_sft_v2 as base
from scripts.build_natural_instructions_memory_v1 import STATE as SOURCE
from scripts.rank_natural_instructions_v26_v1 import STATE as SCORE
from scripts.freeze_v26_release_folds_v1 import STATE as FROZEN

STATE = SOURCE / 'selected-v1'
OUT = base.WORK / 'exports/memory-harness-v2-natural-instructions-selected-v1'


def example_holds():
    examples = {}
    for job in json.loads((SOURCE / 'full-normalize-jobs-v2.json').read_text()):
        source = job['source']
        if base.file_sha(source['local']) != source['sha256']: raise ValueError('NI raw bytes changed')
        task = json.loads(Path(source['local']).read_text())
        for example in task.get('Positive Examples', []):
            if isinstance(example.get('input'), str):
                query = base.sha(base.norm(task['Definition'][0] + '\n\n' + example['input']))
                examples[query] = source['path']
    db = sqlite3.connect(f'file:{SCORE / "candidates.sqlite"}?mode=ro', uri=True)
    matches = [{'query_hash': q, 'task_file': examples[q], 'candidate_answers': n}
               for q, n in db.execute('SELECT query_hash,COUNT(*) FROM candidates GROUP BY query_hash') if q in examples]
    db.close()
    result = {'public_example_query_identities': len(examples),
        'target_queries_also_used_as_worked_example': len(matches),
        'answers': sum(r['candidate_answers'] for r in matches), 'matches': matches,
        'scope': 'All public positive examples in the admitted task files, including those beyond the two rendered examples; conservative exact-query gate only.'}
    path = SOURCE / 'worked-example-target-overlap-v1.json'
    if path.exists() and json.loads(path.read_text()) != result: raise ValueError('example-overlap hold changed')
    if not path.exists(): base.save(path, result)
    return path, {('allenai/natural-instructions', r['query_hash']) for r in matches}


def finalize():
    frozen = json.loads((FROZEN / 'summary.json').read_text())
    if frozen['frozen_database_sha256'] != base.file_sha(FROZEN / 'frozen-aliases.sqlite'):
        raise ValueError('frozen fold aliases changed')
    if not frozen['all_candidate_aliases_included'] or not frozen['all_ancestral_folds_preserved']:
        raise ValueError('incomplete prior fold coverage')
    path, holds = example_holds()
    example_count = len(holds)
    db = sqlite3.connect(f'file:{SCORE / "candidates.sqlite"}?mode=ro', uri=True)
    aliases = [{'query_hash': q, 'complete_inputs': n, 'answers': a} for q, n, a in db.execute(
        'SELECT query_hash,COUNT(DISTINCT input_hash),COUNT(*) FROM candidates GROUP BY query_hash HAVING COUNT(DISTINCT input_hash)>1')]
    db.close()
    holds.update(('allenai/natural-instructions', r['query_hash']) for r in aliases)
    STATE.mkdir(exist_ok=True)
    combined = STATE / 'additional-quality-holds-v1.json'
    base.save(combined, {'worked_example_receipt_sha256': base.file_sha(path),
        'condition_alias_queries': aliases, 'rule': 'Do not count case/space variants with unequal literal conditions as separate tasks or compare their PPL as if conditioned identically.',
        'held_source_queries': sorted(holds), 'all_original_candidate_files_and_scores_preserved': True})
    from scripts import finalize_v26_answer_selection as selector
    selector.SCORE = SCORE; selector.STATE = STATE; selector.OUT = OUT; selector.SECOND = FROZEN
    selector.EXTRA_SOURCE_QUERY_HOLDS = holds; selector.EXTRA_SOURCE_QUERY_HOLD_RECEIPT = combined
    selector.main()
    base.save(STATE / 'release-scope.json', {
        'scope': 'New NI wave only. Append to the verified v26 release after new compilation, first-winner diagnosis and public packaging.',
        'worked_example_target_query_holds': example_count, 'unequal_condition_alias_query_holds': len(aliases),
        'frozen_prior_summary_sha256': base.file_sha(FROZEN / 'summary.json'),
        'first_source_diagnosis_sha256': base.file_sha(SOURCE / 'semantic-audit-v1/summary.json'),
        'normalization_policy_sha256': base.file_sha(SOURCE / 'full-normalization-policy-v2.json'),
        'all_alternative_minima_recomputed': True, 'not_yet_published': True, 'weights_trained': False})


def compile_all(workers):
    if not (STATE / 'release-scope.json').exists(): raise ValueError('NI finalization incomplete')
    base.STATE = STATE; base.OUT = OUT; base.compile_all(workers, False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['example-holds', 'finalize', 'compile'])
    p.add_argument('--workers', type=int, default=16); args = p.parse_args()
    if not 1 <= args.workers <= 32: raise ValueError('CPU worker bound')
    if args.phase == 'example-holds':
        path, holds = example_holds(); print(json.dumps({'held_queries': len(holds), 'receipt': str(path)}), flush=True)
    elif args.phase == 'finalize': finalize()
    else: compile_all(args.workers)
