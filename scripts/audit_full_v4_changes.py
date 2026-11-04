"""Count actual text changes, separately from idempotent skill refresh intent."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from collections import Counter
import json,time,traceback
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.audit_full_v4_source_visibility import verified_file
from barbet_pipeline.full_corpus import STATE,refresh_singleton_skills
from barbet_pipeline.common import read_json,save_new,sha_file,digest

JOBS={}


def initialize():
    global JOBS
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    JOBS={j['path']:j for j in read_json(STATE/'plan.json')['jobs']}


def process(path):
    target=STATE/'actual-changes'/path.name
    if target.exists():return read_json(target)
    receipt=read_json(path);job=JOBS[receipt['source_path']]
    if sha_file(job['local'])!=receipt['source_sha256']:raise ValueError('Original source bytes changed')
    prior=set(job['excluded_indices']);new_holds=set();counts=Counter();by_source=Counter()
    if receipt['counts'].get('quarantined',0):
        item=next(f for f in receipt['files'] if '/dispositions/' in f['remote'])
        with verified_file(item) as handle:
            for batch in pq.ParquetFile(handle).iter_batches(columns=['parent_record_id','status']):
                new_holds.update(r['parent_record_id'] for r in batch.to_pylist() if r['status']=='quarantined')
        if len(new_holds)!=receipt['counts']['quarantined']:raise ValueError('New hold coverage differs')
    offset=0
    for batch in pq.ParquetFile(job['local']).iter_batches(batch_size=512,columns=['record_id','source','messages_json','skills_json','source_refs_json']):
        for index,row in enumerate(batch.to_pylist(),offset):
            if index in prior or row['record_id'] in new_holds:continue
            selection=json.loads(row['source_refs_json']).get('answer_selection',{})
            counts['exported_rows']+=1
            if row['source']=='rajpurkar/squad_v2' and selection.get('competitors')==1:
                counts['natural_view_bytes_changed']+=1;kind='natural_view_changed'
            else:
                first=json.loads(row['messages_json'])[0]
                updated,metadata=refresh_singleton_skills(row,[first])
                if metadata is None:counts['original_input_retained']+=1;kind='original_retained'
                elif updated[0]['content']==first['content']:
                    counts['skill_refresh_already_byte_identical']+=1;kind='skill_refresh_identical'
                else:counts['skill_refresh_bytes_changed']+=1;kind='skill_text_changed'
            by_source[row['source']+'|'+kind]+=1
        offset+=len(batch)
    if counts['exported_rows']!=receipt['counts']['exported']:raise ValueError('Actual change census missed rows')
    if counts['natural_view_bytes_changed']!=receipt['counts']['natural_views_replaced']:raise ValueError('Natural change classification mismatch')
    if counts['skill_refresh_already_byte_identical']+counts['skill_refresh_bytes_changed']!=receipt['counts']['skill_prefills_updated']:
        raise ValueError('Skill refresh classification mismatch')
    counts['actual_serialized_text_changed']=counts['natural_view_bytes_changed']+counts['skill_refresh_bytes_changed']
    counts['actual_serialized_text_identical']=counts['exported_rows']-counts['actual_serialized_text_changed']
    result={'source_receipt_sha256':sha_file(path),'counts':dict(counts),'by_source':dict(by_source),
        'method':'Original source hash and dispositions; natural singleton replacement or exact old/new system comparison. Other messages and tools are frozen by the primary full-row proof.'}
    save_new(target,result);return result


def main():
    finished=set();parts=[]
    with ProcessPoolExecutor(16,initializer=initialize,mp_context=get_context('fork')) as pool:
        while len(finished)<1504:
            pending=[p for p in sorted((STATE/'receipts').glob('*.json')) if p.name not in finished]
            if not pending:time.sleep(5);continue
            futures={pool.submit(process,p):p for p in pending};failures=[]
            for future in as_completed(futures):
                try:result=future.result()
                except Exception:
                    error={'source_receipt':futures[future].name,'traceback':traceback.format_exc()}
                    save_new(STATE/'actual-changes-errors'/futures[future].name,error);failures.append(error)
                    print(json.dumps(error),flush=True);continue
                parts.append(result);finished.add(futures[future].name)
                if len(finished)%128==0:print(json.dumps({'shards':len(finished),'rows':sum(p['counts']['exported_rows'] for p in parts)}),flush=True)
            if failures:raise RuntimeError('Actual change census failed; no complete result claimed')
    counts=Counter();sources=Counter()
    for part in parts:counts.update(part['counts']);sources.update(part['by_source'])
    result={'source_shards':len(parts),'counts':dict(counts),'by_source':dict(sources),'all_exported_rows_classified':True,
        'input_view_changed_field_is_transformation_intent_not_a_byte_difference_proof':True,
        'skill_refresh_can_be_idempotent':True,'new_independent_questions_created_by_refresh':False}
    save_new(STATE/'actual-changes-summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
