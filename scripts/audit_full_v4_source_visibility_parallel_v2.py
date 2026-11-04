"""Bounded parallel read-only source-prefix audit; reuse sealed scan receipts."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from collections import Counter
import json,time,traceback
import pyarrow as pa
from scripts.audit_full_v4_source_visibility_v2 import scan
from barbet_pipeline.full_corpus import STATE
from barbet_pipeline.common import read_json,save_new,sha_file


def initialize():pa.set_cpu_count(1);pa.set_io_thread_count(1)


def process(path):
    target=STATE/'source-visibility'/path.name
    if target.exists():return read_json(target)
    result=scan(read_json(path));result['source_receipt_sha256']=sha_file(path);save_new(target,result)
    return result


def main():
    finished=set();parts=[]
    with ProcessPoolExecutor(8,initializer=initialize,mp_context=get_context('fork')) as pool:
        while len(finished)<1504:
            pending=[p for p in sorted((STATE/'receipts').glob('*.json')) if p.name not in finished]
            if not pending:time.sleep(5);continue
            futures={pool.submit(process,p):p for p in pending}
            failures=[]
            for future in as_completed(futures):
                try:result=future.result()
                except Exception:
                    failure={'source_receipt':futures[future].name,'traceback':traceback.format_exc()}
                    save_new(STATE/'source-visibility-v2-errors'/futures[future].name,failure)
                    failures.append(failure);print(json.dumps(failure),flush=True);continue
                parts.append(result);finished.add(futures[future].name)
                if len(finished)%32==0:
                    print(json.dumps({'source_shards':len(finished),'checked_QA_NI_rows':sum(sum(r['source_counts'].values()) for r in parts)}),flush=True)
            if failures:raise RuntimeError('Source visibility failures preserved; no completed audit claimed')
    counts=Counter();scopes=Counter()
    for part in parts:counts.update(part['source_counts']);scopes.update(part['check_scopes'])
    result={'source_shards':len(parts),'source_counts':dict(counts),'check_scopes':dict(scopes),
        'all_exported_QA_NI_visible_prefixes_reconstructed':True,
        'scope':'source fidelity including actual visible task, evidence and retrieval; not an independent semantic or learnability judgment'}
    save_new(STATE/'source-visibility-summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    assert read_json(STATE/'actual-retrieval-tie-control.json')['passed']
    main()
