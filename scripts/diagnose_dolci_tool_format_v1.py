"""Deterministic source-format diagnosis, without model calls or admission."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pyarrow.parquet as pq
from barbet_inference.memory import canonical
from barbet_inference.sft import Rejected
from scripts import build_memory_sft_v2 as base
from scripts.inventory_dolci_context_v2 import STATE as INVENTORY
from scripts.dolci_tool_projection_v1 import project_tool_messages

STATE = INVENTORY.parent / 'tool-format-diagnosis-v1'
CONTRACT = base.REPO / 'configs/source_tool_contract_holds_v1.json'


def prepare():
    STATE.mkdir(exist_ok=True); (STATE / 'parts').mkdir(exist_ok=True)
    if (STATE / 'plan.json').exists(): return json.loads((STATE / 'plan.json').read_text())
    inventory = json.loads((INVENTORY / 'summary.json').read_text()); pool = {}
    for part_number, part in enumerate(inventory['parts']):
        if base.file_sha(part['part']) != part['sha256']: raise ValueError('inventory bytes changed')
        for row in pq.read_table(part['part'], columns=['row_index', 'query_hash', 'source_dataset']).to_pylist():
            if row['source_dataset'] != 'Dolci Instruct Tool Use': continue
            query = row['query_hash']; rank = base.sha('dolci-tool-format-v1:' + query)
            if query not in pool: pool[query] = (rank, part_number, row['row_index'])
            if len(pool) > 1024:
                keep = sorted(pool, key=lambda q: pool[q][0])[:512]; pool = {q: pool[q] for q in keep}
    chosen = [(q, *value) for q, value in sorted(pool.items(), key=lambda x: x[1][0])[:512]]
    jobs = []
    for part_number, part in enumerate(inventory['parts']):
        rows = [{'query_hash': q, 'row_index': index} for q, rank, number, index in chosen if number == part_number]
        if rows: jobs.append({'source': part['source'], 'part_number': part_number, 'rows': rows})
    plan = {'driver_sha256': base.file_sha(Path(__file__)),
        'projection_sha256': base.file_sha(base.REPO / 'scripts/dolci_tool_projection_v1.py'),
        'literal_parser_sha256': base.file_sha(base.REPO / 'barbet_inference/call_text.py'),
        'source_inventory_sha256': base.file_sha(INVENTORY / 'summary.json'),
        'source_contract_policy_sha256': base.file_sha(CONTRACT), 'jobs': jobs, 'selected_queries': len(chosen),
        'selection': 'Lowest fixed hashes across all explicit Dolci tool-source query identities; one source occurrence per diagnostic query. Not a standard-answer choice.',
        'model_requests': 0, 'accepted_training_rows': 0, 'source_content_license_admitted': False}
    base.save(STATE / 'plan.json', plan); return plan


def job(spec):
    source = spec['source']; path = STATE / 'parts' / f"{spec['part_number']:03d}.json"
    if path.exists(): return json.loads(path.read_text())
    if base.file_sha(source['local']) != source['sha256']: raise ValueError('source bytes changed')
    selected = {r['row_index']: r['query_hash'] for r in spec['rows']}
    contract = json.loads(CONTRACT.read_text()); offset = 0; counts = Counter(); results = []
    for batch in pq.ParquetFile(source['local']).iter_batches(batch_size=1024):
        for raw in batch.to_pylist():
            index = offset; offset += 1
            if index not in selected: continue
            counts['diagnostic_queries'] += 1
            try:
                projected = project_tool_messages(raw['messages'], contract)
                counts['structure_projected'] += 1
                counts['projected_assistant_turns'] += sum(m['role'] == 'assistant' for m in projected['messages'])
                counts['projected_observations'] += sum(m['role'] == 'tool' for m in projected['messages'])
                result = {'status': 'structure_projected_not_semantically_verified',
                    'projected_messages_sha256': base.sha(canonical(projected['messages'])),
                    'explicit_tools_sha256': base.sha(canonical(projected['tools']))}
            except Rejected as exc:
                reason = str(exc); counts['held:' + reason] += 1
                result = {'status': 'format_or_contract_held', 'reason': reason}
            results.append({'source_row_index': index, 'query_hash': selected[index], 'source_id': raw['id'], **result})
    if len(results) != len(selected): raise ValueError('source diagnostic row coverage differs')
    result = {'source_sha256': source['sha256'], 'counts': dict(counts), 'results': results}
    base.save(path, result); return result


def main():
    plan = prepare()
    for key, path in [('driver_sha256', Path(__file__)), ('projection_sha256', base.REPO / 'scripts/dolci_tool_projection_v1.py'),
                      ('literal_parser_sha256', base.REPO / 'barbet_inference/call_text.py'), ('source_contract_policy_sha256', CONTRACT)]:
        if plan[key] != base.file_sha(path): raise ValueError('diagnostic code or contract changed')
    if (STATE / 'summary.json').exists(): raise ValueError('tool format diagnosis already complete')
    counts = Counter(); parts = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for part in pool.map(job, plan['jobs']): counts.update(part['counts']); parts.append(part)
    if counts['diagnostic_queries'] != plan['selected_queries']: raise ValueError('diagnostic query coverage differs')
    result = {'counts': dict(counts), 'parts': parts, 'accepted_training_rows': 0,
        'source_programs_executed': False, 'model_requests': 0, 'source_content_license_admitted': False,
        'scope': '512 deterministically chosen tool-source queries, structural projection only. No semantic accuracy or PPL selection claim.'}
    base.save(STATE / 'summary.json', result); print(json.dumps({k: v for k, v in result.items() if k != 'parts'}), flush=True)


if __name__ == '__main__': main()
