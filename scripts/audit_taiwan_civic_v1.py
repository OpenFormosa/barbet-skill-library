"""Topic triage, not a semantic classifier or a political truth oracle."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scripts import publish_barbet_readiness_v1 as release

STATE = release.parent.base.WORK / 'reports/taiwan-civic-v1'
TAIWAN = r'(?i)(taiwan|formosa|台灣|臺灣|台湾)'
STATUS = r'(?i)(sovereign|independen|unificat|reunificat|self.determin|one.china|part of china|province of china|主權|主权|獨立|独立|統一|统一|自決|自决|一個中國|一个中国|中國的一部分|中国的一部分)'
ABUSE = r'(?i)(torture (is|should be) (good|justified|necessary)|genocide (is|should be) (good|justified)|should (kill|exterminate) all|應該消滅所有|应该消灭所有|酷刑是正當|酷刑是正当)'


def scan(item):
    pa.set_cpu_count(1)
    src = release.parent.parent.DEST / item['path']
    table = pq.read_table(src, columns=['record_id', 'query_hash', 'family_id', 'source', 'messages_json', 'source_refs_json'])
    selected = pc.or_(pc.and_(pc.match_substring_regex(table['messages_json'], TAIWAN),
                             pc.match_substring_regex(table['messages_json'], STATUS)),
                      pc.match_substring_regex(table['messages_json'], ABUSE))
    rows = []
    excluded = set(release.read(release.audit.STATE / 'new-query-holds-v1.json')['query_hashes'])
    structured = 0
    for row in table.filter(selected).to_pylist():
        if row['query_hash'] in excluded: continue
        refs = json.loads(row['source_refs_json'])
        if refs.get('query_schema_id'):
            structured += 1
            continue
        messages = json.loads(row['messages_json'])
        targets = '\n'.join(m.get('content', '') for m in messages if m['role'] == 'assistant' and m.get('loss'))
        target_topic = bool((re.search(TAIWAN, targets) and re.search(STATUS, targets)) or re.search(ABUSE, targets))
        if not target_topic: continue
        rows.append({'path': item['path'], 'record_id': row['record_id'], 'query_hash': row['query_hash'],
                     'family_id': row['family_id'], 'source': row['source'], 'targets': targets,
                     'reason': 'unreviewed_natural_language_sovereignty_or_extreme_abuse_target'})
    return {'scanned': len(table), 'structured_topic_matches': structured, 'rows': rows}


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'existing-topic-triage.json').exists(): raise ValueError('First triage already frozen')
    manifest = release.read(release.STATE / 'plan.json')['base_manifest']
    jobs = [x for x in manifest if '/records/' in x['path']]
    rows = []; counts = Counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for n, part in enumerate(pool.map(scan, jobs), 1):
            rows.extend(part.pop('rows')); counts.update(part)
            if n % 100 == 0 or n == len(jobs):
                print(json.dumps({'shards': n, 'total': len(jobs), 'candidate_targets': len(rows)}), flush=True)
    release.save(STATE / 'existing-topic-targets.private.json', rows)
    holds = sorted({r['query_hash'] for r in rows})
    release.save(STATE / 'existing-topic-triage.json', {'driver_sha256': release.sha(Path(__file__)),
        'counts': dict(counts), 'candidate_records': len(rows), 'held_query_hashes': holds,
        'source_counts': dict(Counter(r['source'] for r in rows)),
        'disposition': 'Conservative whole-query quarantine pending contextual policy review; a topic match does not prove a false or objectionable answer.',
        'finite_structured_queries_not_world_truth_endorsements': True,
        'exhaustive_semantic_or_political_alignment_audit': False,
        'all_parent_rows_scanned_by_lexical_triage': True,
        'query_wide_no_runner_up_substitution': True})
    print(json.dumps({'candidate_records': len(rows), 'held_queries': len(holds), 'counts': dict(counts)}), flush=True)


if __name__ == '__main__': main()
