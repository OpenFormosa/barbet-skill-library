"""Exhaustive source-visible QA/NI audit while complete shards become available."""
from pathlib import Path
from contextlib import contextmanager
from collections import Counter
import hashlib,json,time,tempfile
from urllib import request,parse
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json,save_new,sha_file
from barbet_pipeline.full_corpus import STATE
from barbet_pipeline.full_publication import REPO,BRANCH
from barbet_pipeline.full_source_visibility import SOURCES
from barbet_pipeline.full_source_visibility_v2 import check_source_visibility


@contextmanager
def verified_file(item):
    with tempfile.TemporaryDirectory(dir=STATE,prefix='visible-audit-download-') as temporary:
        try:handle=Path(item['local']).open('rb')
        except FileNotFoundError:
            path=Path(temporary)/'records.parquet'
            url=f'https://huggingface.co/datasets/{REPO}/resolve/{BRANCH}/'+parse.quote(item['remote'],safe='/')
            with request.urlopen(url,timeout=180) as stream,path.open('xb') as dest:
                for chunk in iter(lambda:stream.read(1024*1024),b''):dest.write(chunk)
            handle=path.open('rb')
        with handle:
            h=hashlib.sha256()
            for chunk in iter(lambda:handle.read(8*1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=item['sha256']:raise ValueError('Audit input bytes changed')
            handle.seek(0);yield handle


def scan(receipt):
    expected={s:sum(receipt['by_source'].get(t+':'+s,0) for t in ['verified','source_reference']) for s in SOURCES}
    counts=Counter();scopes=Counter()
    if sum(expected.values()):
        for f in receipt['files']:
            if not any(f['remote'].startswith('full-v4/'+t+'/records/') for t in ['verified','source_reference']):continue
            with verified_file(f) as handle:
                for batch in pq.ParquetFile(handle).iter_batches(batch_size=256,columns=['source','source_refs_json','messages_json','memory_json','tools_json','natural_answer_rewritten','memory_placement']):
                    for row in batch.to_pylist():
                        scope=check_source_visibility(row)
                        if scope:counts[row['source']]+=1;scopes[scope]+=1
    if any(counts[s]!=v for s,v in expected.items()):raise ValueError('Source visibility audit coverage incomplete')
    return {'source_counts':dict(counts),'check_scopes':dict(scopes),'all_expected_rows_checked':True,'checker_version':'visible-source-v2-exact-fsum-ties'}


def main():
    finished=set();parts=[]
    while len(finished)<1504:
        pending=[p for p in sorted((STATE/'receipts').glob('*.json')) if p.name not in finished]
        if not pending:time.sleep(5);continue
        for path in pending:
            target=STATE/'source-visibility'/path.name
            if target.exists():result=read_json(target)
            else:
                result=scan(read_json(path));result['source_receipt_sha256']=sha_file(path);save_new(target,result)
            parts.append(result);finished.add(path.name)
        print(json.dumps({'source_shards':len(finished),'checked_QA_NI_rows':sum(sum(r['source_counts'].values()) for r in parts)}),flush=True)
    counts=Counter();scopes=Counter()
    for part in parts:counts.update(part['source_counts']);scopes.update(part['check_scopes'])
    result={'source_shards':len(parts),'source_counts':dict(counts),'check_scopes':dict(scopes),'all_exported_QA_NI_visible_prefixes_reconstructed':True,
            'scope':'source fidelity including actual visible task, evidence and retrieval; not an independent semantic or learnability judgment'}
    save_new(STATE/'source-visibility-summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
