"""Exhaustive finite-graph visible operation/evidence audit, eight read workers."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from collections import Counter
import json,time,traceback
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.audit_full_v4_source_visibility import verified_file
from barbet_pipeline.full_corpus import STATE
from barbet_pipeline.full_graph_visibility import check_graph_visibility
from barbet_pipeline.common import read_json,save_new,sha_file


def initialize():pa.set_cpu_count(1);pa.set_io_thread_count(1)


def process(path):
    target=STATE/'graph-visibility'/path.name
    if target.exists():return read_json(target)
    receipt=read_json(path);counts=Counter();shapes=Counter()
    expected={k[len('verified:'):]:v for k,v in receipt['by_source'].items() if k.startswith(('verified:Wikidata5M/','verified:ConceptNet/'))}
    if sum(expected.values()):
        for f in receipt['files']:
            if not f['remote'].startswith('full-v4/verified/records/'):continue
            with verified_file(f) as handle:
                for batch in pq.ParquetFile(handle).iter_batches(batch_size=256,columns=['source','source_refs_json','memory_json','messages_json','tools_json','memory_placement']):
                    for row in batch.to_pylist():
                        result=check_graph_visibility(row)
                        if result:
                            counts[result['source']]+=1
                            shapes['|'.join(result[k] for k in ['operation','path_lengths','memory_placement'])]+=1
    if dict(counts)!=expected:raise ValueError('Graph visible-input audit coverage differs')
    result={'source_receipt_sha256':sha_file(path),'source_counts':dict(counts),'semantic_shape_counts':dict(shapes),
            'actual_visible_question_and_evidence_checked':True}
    save_new(target,result);return result


def main():
    finished=set();parts=[]
    with ProcessPoolExecutor(8,initializer=initialize,mp_context=get_context('fork')) as pool:
        while len(finished)<1504:
            pending=[p for p in sorted((STATE/'receipts').glob('*.json')) if p.name not in finished]
            if not pending:time.sleep(5);continue
            futures={pool.submit(process,p):p for p in pending};failures=[]
            for future in as_completed(futures):
                try:result=future.result()
                except Exception:
                    error={'source_receipt':futures[future].name,'traceback':traceback.format_exc()}
                    save_new(STATE/'graph-visibility-errors'/futures[future].name,error);failures.append(error)
                    print(json.dumps(error),flush=True);continue
                parts.append(result);finished.add(futures[future].name)
                if len(finished)%64==0:print(json.dumps({'shards':len(finished),'graph_rows':sum(sum(p['source_counts'].values()) for p in parts)}),flush=True)
            if failures:raise RuntimeError('Graph visible-input audit failed; original outputs preserved')
    counts=Counter();shapes=Counter()
    for part in parts:counts.update(part['source_counts']);shapes.update(part['semantic_shape_counts'])
    result={'source_shards':len(parts),'source_counts':dict(counts),'semantic_shape_counts':dict(shapes),
        'all_exported_graph_queries_and_visible_state_checked':True,
        'scope':'explicit graph IDs, operations and finite visible state; display labels and world facts are not independently certified'}
    save_new(STATE/'graph-visibility-summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    assert read_json(STATE/'actual-graph-visibility-controls.json')['passed']
    main()
