"""Prove disjoint lexical singleton components against all immutable ancestors."""
from collections import Counter
import json
from pathlib import Path
import sqlite3

from scripts import build_memory_sft_v2 as base
from scripts import normalize_conceptnet_relations_v1 as source
from scripts.index_conceptnet_relations_v1 import STATE as SCORE,reserve

STATE=source.STATE/'selected-v1'
OUT=base.WORK/'exports/memory-harness-v2-conceptnet-relations-selected-v1'


def main():
    STATE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    if (STATE/'selection.sqlite').exists():raise ValueError('singleton finalization already started; preserve first state')
    reserve('singleton-finalize',1024**3)
    index=json.loads((SCORE/'candidate-index-summary.json').read_text());guard=json.loads((SCORE/'episode-identity-v1/summary.json').read_text())
    if index['competing_groups'] or index['answers_to_score'] not in (None,0):
        raise ValueError('alternative answers require complete frozen native-v26 scoring, not singleton selection')
    if index['distinct_answers']!=index['input_groups'] or index['counts']['eligible']!=index['distinct_answers']:
        raise ValueError('singleton cardinality differs')
    if guard['status']!='COMPLETE' or not guard['all_original_candidates_covered']:raise ValueError('complete episode guard required')
    normalized=json.loads((source.STATE/'normalization-policy.json').read_text())
    for key,path in [('prior_frozen_aliases_sha256',source.PRIOR_FROZEN/'frozen-aliases.sqlite'),
        ('prior_graph_candidates_sha256',source.GRAPH_INDEX/'candidates.sqlite'),
        ('prior_graph_selection_sha256',source.GRAPH_SELECTED/'ppl-score-binding.json')]:
        if base.file_sha(path)!=normalized[key]:raise ValueError('immutable prior identity source changed')
    graph_policy=json.loads((source.GRAPH_SELECTED/'selection-policy.json').read_text())
    if not graph_policy['all_source_families_questions_and_inputs_unique'] or graph_policy['ancestral_alias_intersections']:
        raise ValueError('prior graph component rule no longer applicable')
    db=sqlite3.connect(f'file:{SCORE / "candidates.sqlite"}?mode=ro',uri=True)
    if db.execute('SELECT 1 FROM candidates WHERE source_family_id!=template_family_id LIMIT 1').fetchone():
        raise ValueError('source template families require generic component resolution')
    unique={key:db.execute(f'SELECT COUNT(DISTINCT {key}) FROM candidates').fetchone()[0]
        for key in ['record_id','query_hash','source_family_id']}
    if any(n!=index['distinct_answers'] for n in unique.values()):raise ValueError('cross-anchor identity reuse requires general resolution')
    nodes=set()
    for family,query,condition in db.execute('SELECT source_family_id,query_hash,input_hash FROM candidates'):
        nodes.update(('family:'+family,'query:'+query,'input:'+condition))
    overlaps=[];alias_count=0;graph_count=0
    old=sqlite3.connect(f'file:{source.PRIOR_FROZEN / "frozen-aliases.sqlite"}?mode=ro',uri=True)
    for table in ['aliases','conflicts']:
        for node, in old.execute('SELECT node FROM '+table):
            alias_count+=1
            if node in nodes:overlaps.append(node)
    old.close()
    # Include all graph candidates, not just eventual public winners, and the
    # final graph family alias prescribed by its already-verified component rule.
    graph=sqlite3.connect(f'file:{source.GRAPH_INDEX / "candidates.sqlite"}?mode=ro',uri=True)
    for family,query,condition in graph.execute('SELECT source_family_id,query_hash,input_hash FROM candidates'):
        graph_count+=1
        for node in ('family:'+family,'family:'+base.sha('family:'+family),'query:'+query,'input:'+condition):
            if node in nodes:overlaps.append(node)
    graph.close();node_count=len(nodes);del nodes
    if overlaps:
        base.save(STATE/'generic-component-resolution-required.json',{'shared_aliases':overlaps})
        raise ValueError('prior alias overlap; do not assign a new fold')
    binding={'driver_sha256':base.file_sha(Path(__file__)),
        'candidate_index_summary_sha256':base.file_sha(SCORE/'candidate-index-summary.json'),
        'candidate_database_sha256':base.file_sha(SCORE/'candidates.sqlite'),
        'episode_guard_summary_sha256':base.file_sha(SCORE/'episode-identity-v1/summary.json'),
        'episode_guard_database_sha256':base.file_sha(SCORE/'episode-identity-v1/identity.sqlite'),
        'source_normalization_summary_sha256':base.file_sha(source.STATE/'normalize-summary.json'),
        'prior_frozen_aliases_sha256':normalized['prior_frozen_aliases_sha256'],
        'prior_graph_candidates_sha256':normalized['prior_graph_candidates_sha256'],
        'prior_graph_selection_sha256':normalized['prior_graph_selection_sha256'],
        'prior_aliases_and_conflicts_scanned':alias_count,'all_prior_graph_candidates_scanned':graph_count,
        'ancestral_alias_intersections':0,'source_component_nodes':node_count,
        'all_source_families_questions_and_inputs_unique':True,
        'same_general_component_fold_rule':'family = sha256(family:<source-family-hash>); validation iff first 12 hex digits modulo 100 < 2',
        'prior_identity_databases_not_modified_or_duplicated':True,'all_singletons_unscored':True,
        'model_requests':0,'invented_PPL_values':False,'any_alternatives_require_native_v26_ppl':True}
    base.save(STATE/'selection-policy.json',binding)
    held=sqlite3.connect(f'file:{SCORE / "episode-identity-v1/identity.sqlite"}?mode=ro',uri=True)
    held_ids={r[0] for r in held.execute('SELECT candidate_identity FROM holds')};held.close()
    out=sqlite3.connect(STATE/'selection.sqlite')
    out.execute('CREATE TABLE selected(candidate_identity TEXT PRIMARY KEY,record_id TEXT,family_id TEXT,split TEXT,file_id INTEGER)')
    counts=Counter();files=Counter();buffer=[]
    for cid,rid,family,file_id in db.execute('SELECT candidate_identity,record_id,source_family_id,file_id FROM candidates ORDER BY file_id,offset'):
        if cid in held_ids:counts['excluded_episode_identity']+=1;continue
        final=base.sha('family:'+family);split='validation' if int(final[:12],16)%100<2 else 'train'
        buffer.append((cid,rid,final,split,file_id));counts['selected']+=1;counts['single_eligible_answer']+=1
        counts['split:'+split]+=1;files[file_id]+=1
        if len(buffer)>=10000:out.executemany('INSERT INTO selected VALUES(?,?,?,?,?)',buffer);out.commit();buffer=[]
    out.executemany('INSERT INTO selected VALUES(?,?,?,?,?)',buffer);out.commit()
    out.execute('CREATE INDEX by_source_file ON selected(file_id)');out.commit();out.close();db.close()
    manifest=json.loads((SCORE/'candidate-input-files.json').read_text())
    jobs=[{**f,'selected_rows':files[f['file_id']]} for f in manifest if files[f['file_id']]];counts['shards']=len(jobs)
    base.save(STATE/'compile-jobs.json',jobs);base.save(STATE/'dedup-summary.json',dict(counts))
    base.save(STATE/'ppl-score-binding.json',{'selection_policy_sha256':base.file_sha(STATE/'selection-policy.json'),
        'selection_database_sha256':base.file_sha(STATE/'selection.sqlite'),
        'native_v26_policy_sha256':base.file_sha(base.REPO/'configs/memory_sft_v2_v26_selection.json'),
        'method':'only_eligible_answer','unscored_singletons':counts['selected'],'competing_groups':0,
        'model_requests':0,'measured_ppl_rows':0,'no_likelihood_or_quality_improvement_claim':True})
    base.save(STATE/'release-scope.json',{'all_frozen_folds_preserved':True,'all_first_source_holds_preserved':True,
        'one_task_per_base_concept':True,'source_URI_variants_not_independent_questions':True,
        'exact_target_not_teacher_prose':True,'selected_rows':counts['selected'],'not_yet_published':True,
        'all_records_semantically_verified':False,'model_weights_trained':False,'source_programs_executed':False})
    print(json.dumps(dict(counts)),flush=True)


if __name__=='__main__':main()
