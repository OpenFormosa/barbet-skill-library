"""Confirm singleton identity and folds without copying full candidate text.

An alternative-answer group stops this path: it must use frozen native v26 PPL.
This optimization applies only after proving unique source families, questions
and input groups and no intersection with any frozen ancestral alias.
"""
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3

from scripts import build_memory_sft_v2 as base
from scripts import normalize_wikidata_relations_v1 as source
from scripts.index_wikidata_relations_v1 import STATE as SCORE, FROZEN

STATE=source.STATE/'selected-v1'
OUT=base.WORK/'exports/memory-harness-v2-wikidata-relations-selected-v1'


def main():
    STATE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    if (STATE/'selection.sqlite').exists():raise ValueError('singleton finalization already started; preserve first state')
    resources=json.loads(source.RESOURCES.read_text())['resources'];reserve=5*1024**3
    if source.processed_bytes()+reserve>resources['new_processed_cap_bytes'] or shutil.disk_usage(base.WORK).free<resources['minimum_free_disk_bytes']+reserve:
        raise ValueError('singleton metadata stage exceeds resources')
    index=json.loads((SCORE/'candidate-index-summary.json').read_text())
    guard=json.loads((SCORE/'episode-identity-v1/summary.json').read_text())
    frozen=json.loads((FROZEN/'summary.json').read_text())
    if index['competing_groups'] or index['answers_to_score'] not in (None,0):
        raise ValueError('answer alternatives exist: require complete frozen native-v26 comparison, not singleton selection')
    if index['distinct_answers']!=index['input_groups'] or index['counts']['eligible']!=index['distinct_answers']:
        raise ValueError('source singleton identity count differs')
    if guard['status']!='COMPLETE' or not guard['all_original_candidates_covered']:
        raise ValueError('complete episode identity guard required')
    if base.file_sha(FROZEN/'frozen-aliases.sqlite')!=frozen['frozen_database_sha256']:
        raise ValueError('prior complete family aliases changed')
    db=sqlite3.connect(f'file:{SCORE / "candidates.sqlite"}?mode=ro',uri=True)
    if db.execute('SELECT 1 FROM candidates WHERE source_family_id!=template_family_id LIMIT 1').fetchone():
        raise ValueError('singleton family fast path does not apply')
    unique={k:db.execute(f'SELECT COUNT(DISTINCT {k}) FROM candidates').fetchone()[0] for k in ['record_id','query_hash','source_family_id']}
    if any(v!=index['distinct_answers'] for v in unique.values()):
        raise ValueError('cross-anchor identity reuse requires generic component resolution')
    # Three disjoint node types for every independent anchor. This is the same
    # family/query/input component definition as the general selector.
    nodes=set()
    for family,query,condition in db.execute('SELECT source_family_id,query_hash,input_hash FROM candidates'):
        nodes.update(('family:'+family,'query:'+query,'input:'+condition))
    old=sqlite3.connect(f'file:{FROZEN / "frozen-aliases.sqlite"}?mode=ro',uri=True);overlaps=[]
    for node, in old.execute('SELECT node FROM aliases'):
        if node in nodes:overlaps.append(node)
    for node, in old.execute('SELECT node FROM conflicts'):
        if node in nodes:overlaps.append(node)
    old.close();node_count=len(nodes);del nodes
    if overlaps:
        base.save(STATE/'generic_component_resolution_required.json',{'shared_aliases':overlaps})
        raise ValueError('frozen alias overlap requires generic resolution; do not assign a new fold')
    binding={'driver_sha256':base.file_sha(Path(__file__)),
        'complete_index_summary_sha256':base.file_sha(SCORE/'candidate-index-summary.json'),
        'complete_candidate_database_sha256':base.file_sha(SCORE/'candidates.sqlite'),
        'episode_guard_summary_sha256':base.file_sha(SCORE/'episode-identity-v1/summary.json'),
        'episode_guard_database_sha256':base.file_sha(SCORE/'episode-identity-v1/identity.sqlite'),
        'frozen_prior_summary_sha256':base.file_sha(FROZEN/'summary.json'),
        'source_normalization_summary_sha256':base.file_sha(source.STATE/'normalize-summary.json'),
        'all_source_families_questions_and_inputs_unique':True,'ancestral_alias_intersections':0,
        'source_component_nodes':node_count,'same_general_component_fold_rule':'family = sha256(family:<source-family-hash>); validation iff first 12 hex digits modulo 100 < 2',
        'full_candidate_messages_not_copied':True,'all_singletons_unscored':True,
        'model_requests':0,'ppl_values_invented':False,'any_alternatives_would_require_native_v26_ppl':True}
    base.save(STATE/'selection-policy.json',binding)
    held=sqlite3.connect(f'file:{SCORE / "episode-identity-v1/identity.sqlite"}?mode=ro',uri=True)
    held_ids={r[0] for r in held.execute('SELECT candidate_identity FROM holds')};held.close()
    out=sqlite3.connect(STATE/'selection.sqlite')
    out.execute('CREATE TABLE selected(candidate_identity TEXT PRIMARY KEY,record_id TEXT,family_id TEXT,split TEXT,file_id INTEGER)')
    counts=Counter();file_counts=Counter();buffer=[]
    for cid,rid,family,file_id in db.execute('SELECT candidate_identity,record_id,source_family_id,file_id FROM candidates ORDER BY file_id,offset'):
        if cid in held_ids:counts['excluded_episode_identity']+=1;continue
        final_family=base.sha('family:'+family);split='validation' if int(final_family[:12],16)%100<2 else 'train'
        buffer.append((cid,rid,final_family,split,file_id));counts['selected']+=1;counts['single_eligible_answer']+=1
        counts['split:'+split]+=1;file_counts[file_id]+=1
        if len(buffer)>=10000:out.executemany('INSERT INTO selected VALUES(?,?,?,?,?)',buffer);out.commit();buffer=[]
    out.executemany('INSERT INTO selected VALUES(?,?,?,?,?)',buffer);out.commit()
    out.execute('CREATE INDEX by_source_file ON selected(file_id)');out.commit();out.close();db.close()
    files=json.loads((SCORE/'candidate-input-files.json').read_text())
    jobs=[{**f,'selected_rows':file_counts[f['file_id']]} for f in files if file_counts[f['file_id']]]
    counts['shards']=len(jobs);base.save(STATE/'compile-jobs.json',jobs);base.save(STATE/'dedup-summary.json',dict(counts))
    base.save(STATE/'ppl-score-binding.json',{'selection_policy_sha256':base.file_sha(STATE/'selection-policy.json'),
        'selection_database_sha256':base.file_sha(STATE/'selection.sqlite'),
        'native_v26_policy_sha256':base.file_sha(base.REPO/'configs/memory_sft_v2_v26_selection.json'),
        'method':'only_eligible_answer','unscored_singletons':counts['selected'],'competing_groups':0,
        'model_requests':0,'measured_ppl_rows':0,'no_likelihood_or_quality_improvement_claim':True})
    base.save(STATE/'release-scope.json',{'all_frozen_folds_preserved':True,'first_source_holds_preserved':True,
        'no_duplicate_memory_placement_views':True,'one_source_anchor_per_selected_question':True,
        'exact_procedural_target_not_teacher_prose':True,'selected_rows':counts['selected'],
        'not_yet_published':True,'model_weights_trained':False,'source_programs_executed':False})
    print(json.dumps(dict(counts)),flush=True)


if __name__=='__main__':main()
