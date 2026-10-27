"""Full source-specific scan for completion claims after a terminal traceback."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from barbet_inference.dataset_quality import terminal_completion_hold
from scripts.build_v26_complete_release import STATE
from scripts import build_memory_sft_v2 as base


def job(path):
    receipt = STATE / 'terminal-completion-audit-v1/parts' / (path.stem + '.json')
    if receipt.exists(): return json.loads(receipt.read_text())
    digest = hashlib.sha256(); counts = Counter(); held = []
    for line in path.open('rb'):
        digest.update(line); row = json.loads(line); counts['seen'] += 1
        if row['upstream_source'] == 'nvidia/Nemotron-Terminal-Corpus': counts['terminal_source'] += 1
        reason = terminal_completion_hold(row)
        if reason:
            counts[reason] += 1
            held.append({'record_id': row['record_id'], 'query_hash': row['query_hash'],
                         'source_file_sha256': row['source_refs']['file_sha256'], 'reason': reason})
    result = {'selected_file': path.name, 'selected_file_sha256': digest.hexdigest(),
              'counts': dict(counts), 'held': held}
    base.save(receipt, result); return result


def main():
    if not (STATE / 'dedup-summary.json').exists(): raise ValueError('selection incomplete')
    target = STATE / 'terminal-completion-audit-v1'; target.mkdir(parents=True, exist_ok=True)
    if (target / 'summary.json').exists(): raise ValueError('terminal audit already complete')
    counts = Counter(); held = []; bindings = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for result in pool.map(job, sorted((STATE / 'selected').glob('*.jsonl'))):
            counts.update(result['counts']); held.extend(result['held'])
            bindings.append({k: result[k] for k in ['selected_file', 'selected_file_sha256']})
    if counts['seen'] != json.loads((STATE / 'dedup-summary.json').read_text())['selected']:
        raise ValueError('terminal scan coverage incomplete')
    base.save(target / 'summary.json', {'counts': dict(counts), 'held': held, 'selected_shard_bindings': bindings,
        'gate_code_sha256': base.file_sha(base.REPO / 'barbet_inference/dataset_quality.py'),
        'origin': 'First winner diagnostic found a terminal completion claim after a SyntaxError in the last recorded observation.',
        'scope': 'Exact source-specific JSON task_complete true plus an explicit exception line in the last run_terminal observation. A conservative unresolved concern, not proof that every printed error invalidates the task.',
        'does_not_prove_older_errors_were_resolved': True, 'source_programs_executed': False,
        'existing_first_negative_holds_preserved': True})
    print(json.dumps({'counts': dict(counts), 'held': len(held)}), flush=True)


if __name__ == '__main__': main()
