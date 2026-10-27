"""Rebuild cumulative candidates from complete minimum-PPL decisions.

Old release files remain immutable. Frozen old folds and all first holds survive.
"""
from collections import Counter
import json
from pathlib import Path
import sqlite3
from scripts import build_memory_sft_v2 as base
from scripts.build_memory_sft_wave2 import STATE as SECOND
from scripts.prepare_v26_answer_candidates import STATE as SCORE,load_candidate
from barbet_inference.answer_selection import select_minimum_ppl
from barbet_inference.memory import canonical

STATE=base.WORK/'reports/memory-harness-v2-v26-selected'
OUT=base.WORK/'exports/memory-harness-v2-v26-selected'
EXTRA_SOURCE_QUERY_HOLDS=set()
EXTRA_SOURCE_QUERY_HOLD_RECEIPT=None
ROWS_PER_SHARD=2000


def eligible_source_conditions(pointers):
    # A missing-premise condition in one source must not invalidate a separate
    # document-grounded condition merely because its short question is equal.
    return [p for p in pointers if (p['source'],p['query_hash']) not in EXTRA_SOURCE_QUERY_HOLDS]


def merge_decisions(db):
    folder=SCORE/'scores/canonical-v2';protocols=set();receipts=[]
    for bucket in range(8):
        receipt=json.loads((folder/f'bucket-{bucket}.summary.json').read_text())
        if receipt['status']!='BUCKET_COMPLETE' or receipt['limited_pilot']:raise ValueError('unfinished full scoring bucket')
        protocols.add(receipt['protocol_sha256']);receipts.append(receipt)
    if len(protocols)!=1:raise ValueError('incomparable scoring protocols')
    db.execute('CREATE TABLE decisions(input_hash TEXT PRIMARY KEY,status TEXT,winner TEXT,report_json TEXT)')
    db.execute('CREATE TABLE scores(candidate_identity TEXT PRIMARY KEY,input_hash TEXT,record_id TEXT,score_json TEXT)')
    bindings=[]
    for bucket in range(8):
        path=folder/f'bucket-{bucket}.sqlite';bindings.append({'bucket':bucket,'sha256':base.file_sha(path)})
        db.execute('ATTACH DATABASE ? AS shard',(str(path),))
        db.execute('INSERT INTO decisions SELECT * FROM shard.decisions')
        db.execute('INSERT INTO scores SELECT * FROM shard.scores');db.commit();db.execute('DETACH DATABASE shard')
    expected=db.execute('SELECT COUNT(*) FROM candidates.groups WHERE candidates>1').fetchone()[0]
    actual=db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]
    if actual!=expected:raise ValueError('missing or extra group decisions')
    if db.execute('SELECT COUNT(*) FROM decisions d LEFT JOIN candidates.groups g USING(input_hash) WHERE g.candidates IS NULL OR g.candidates<2').fetchone()[0]:raise ValueError('decision input mismatch')
    db.execute('CREATE INDEX scored_groups ON scores(input_hash)');db.commit()
    for group in db.execute("SELECT * FROM decisions WHERE status='selected'").fetchall():
        scores=[json.loads(x[0]) for x in db.execute('SELECT score_json FROM scores WHERE input_hash=?',(group['input_hash'],))]
        count=db.execute('SELECT candidates FROM candidates.groups WHERE input_hash=?',(group['input_hash'],)).fetchone()[0]
        if len(scores)!=count:raise ValueError('selected group has incomplete competing scores')
        best=select_minimum_ppl(scores)
        if best['candidate_identity']!=group['winner']:raise ValueError('winner is not minimum eligible answer NLL')
    base.save(STATE/'ppl-score-binding.json',{'protocol_sha256':next(iter(protocols)),'buckets':bindings,'group_decisions':actual,
        'all_minima_independently_recomputed_from_saved_nll':True,'singletons_not_claimed_scored':True})


def strict_condition(row):
    # IDs inside tool observations are part of the actual native conditioning
    # text. If only semantic aliasing grouped different byte conditions, hold
    # that group rather than claim a fair fixed-input PPL comparison.
    inputs=[{k:v for k,v in m.items() if k!='loss'} for m in row['messages']
            if m['role']!='assistant' or not m.get('loss',True)]
    return base.sha(canonical({'inputs':inputs,'tools':row['tools']}))


def main():
    if not 1<=ROWS_PER_SHARD<=20000:raise ValueError('selected shard row bound')
    STATE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (STATE/'dedup-summary.json').exists():raise ValueError('selection already frozen')
    db=sqlite3.connect(STATE/'selection.sqlite');db.row_factory=sqlite3.Row
    if db.execute("SELECT name FROM sqlite_master WHERE name='decisions'").fetchone():raise ValueError('partial finalization exists; inspect rather than overwrite')
    db.execute('ATTACH DATABASE ? AS candidates',(str(SCORE/'candidates.sqlite'),))
    identity=json.loads((SCORE/'episode-identity-v1/summary.json').read_text())
    if identity['status']!='COMPLETE' or not identity['all_original_candidates_covered']:raise ValueError('episode identity guard incomplete')
    db.execute('ATTACH DATABASE ? AS identity',(str(SCORE/'episode-identity-v1/identity.sqlite'),))
    merge_decisions(db)
    parent={}
    def find(x):
        parent.setdefault(x,x);root=x
        while parent[root]!=root:root=parent[root]
        while parent[x]!=x:old=parent[x];parent[x]=root;x=old
        return root
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for r in db.execute('SELECT source_family_id,template_family_id,query_hash,input_hash FROM candidates.candidates'):
        node='family:'+r['source_family_id']
        for other in ['family:'+r['template_family_id'],'query:'+r['query_hash'],'input:'+r['input_hash']]:union(node,other)
    frozen=sqlite3.connect(f'file:{SECOND / "frozen-aliases.sqlite"}?mode=ro',uri=True)
    ties={};conflicts=set()
    for node,family,split in frozen.execute('SELECT node,family_id,split FROM aliases'):
        if node not in parent:continue
        root=find(node)
        if root in ties and ties[root][1]!=split:conflicts.add(root)
        else:ties[root]=min(ties.get(root,(family,split)),(family,split))
    for node, in frozen.execute('SELECT node FROM conflicts'):
        if node in parent:conflicts.add(find(node))
    frozen.close();files=json.loads((SCORE/'candidate-input-files.json').read_text())
    counts=Counter();writer=None;shard=-1;size=0;selected=set()
    (STATE/'selected').mkdir(exist_ok=True)
    # Group ordering is stable and independent of model score or fold.
    for n,g in enumerate(db.execute('SELECT * FROM candidates.groups ORDER BY input_hash'),1):
        pointers=list(db.execute('SELECT c.* FROM candidates.candidates c WHERE input_hash=? AND NOT EXISTS(SELECT 1 FROM identity.holds h WHERE h.candidate_identity=c.candidate_identity) ORDER BY candidate_identity',(g['input_hash'],)))
        counts['excluded_prefix_or_ambiguous_answers']+=g['candidates']-len(pointers)
        eligible=eligible_source_conditions(pointers)
        counts['excluded_source_premise_answers']+=len(pointers)-len(eligible)
        pointers=eligible
        if not pointers:counts['held_no_complete_unambiguous_answer']+=1;continue
        root=find('input:'+g['input_hash'])
        if root in conflicts:counts['held_frozen_fold_bridge']+=1;continue
        decision=None
        if g['candidates']>1:
            decision=db.execute('SELECT * FROM decisions WHERE input_hash=?',(g['input_hash'],)).fetchone()
            if decision['status']!='selected':counts['held_incomplete_score_group']+=1;continue
            conditions={strict_condition(load_candidate(p,files)) for p in pointers}
            if len(conditions)!=1:counts['held_actual_condition_bytes_differ']+=1;continue
            eligible_scores=[json.loads(db.execute('SELECT score_json FROM scores WHERE candidate_identity=?',(p['candidate_identity'],)).fetchone()[0]) for p in pointers]
            best=select_minimum_ppl(eligible_scores)
            pointer=next(p for p in pointers if p['candidate_identity']==best['candidate_identity'])
        else:pointer=pointers[0]
        row=load_candidate(pointer,files)
        if row['record_id'] in selected:raise ValueError('duplicate selected record identity')
        selected.add(row['record_id'])
        if root in ties:row['family_id'],row['split']=ties[root]
        else:
            row['family_id']=base.sha(root);row['split']='validation' if int(row['family_id'][:12],16)%100<2 else 'train'
        if decision:
            score=json.loads(db.execute('SELECT score_json FROM scores WHERE candidate_identity=?',(pointer['candidate_identity'],)).fetchone()[0])
            row['source_refs']['answer_selection']={'method':'minimum_v26_conditional_answer_ppl','competitors':len(pointers),'original_candidate_count':g['candidates'],
                'input_hash':g['input_hash'],'protocol_sha256':score['protocol_sha256'],'mean_nll':score['mean_nll'],
                'ppl':score['ppl'],'answer_tokens':score['answer_tokens'],'scoring_text_sha256':score['text_sha256'],
                'model_revision':'67131067ff1825179d5e38e91d499159669e9c66','not_a_correctness_proof':True}
            counts['minimum_ppl_selected']+=1
        else:
            row['source_refs']['answer_selection']={'method':'only_eligible_answer','competitors':1,'ppl_measured':False}
            counts['single_eligible_answer']+=1
        if writer is None or size>=ROWS_PER_SHARD:
            if writer:writer.close()
            shard+=1;size=0;writer=(STATE/'selected'/f'{shard:05d}.jsonl').open('w')
        writer.write(canonical(row)+'\n');size+=1;counts['selected']+=1
        counts['domain:'+row['domain']]+=1;counts['split:'+row['split']]+=1
        if n%100000==0:print(json.dumps({'input_groups_processed':n,'counts':dict(counts)}),flush=True)
    if writer:writer.close()
    counts['shards']=shard+1
    base.save(STATE/'dedup-summary.json',dict(counts))
    base.save(STATE/'selection-policy.json',{'version':'v26-minimum-answer-ppl-cumulative-0003','same_actual_condition_bytes_required':True,
        'rows_per_selected_shard':ROWS_PER_SHARD,
        'all_frozen_folds_preserved':True,'cross_split_component_bridges_held':len(conflicts),'all_first_negative_holds_preserved':True,
        'target_new_episodes':10000000,'goal_complete':False,'public_ppl_selected_data_not_yet_published':True,
        'additional_source_premise_hold_receipt_sha256':base.file_sha(EXTRA_SOURCE_QUERY_HOLD_RECEIPT) if EXTRA_SOURCE_QUERY_HOLD_RECEIPT else None})
    db.close();print(json.dumps(dict(counts)),flush=True)


if __name__=='__main__':main()
