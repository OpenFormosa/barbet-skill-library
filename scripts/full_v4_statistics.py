"""Measure all exported rows as immutable worker receipts become available."""
from pathlib import Path
from collections import Counter,defaultdict
import json,time,tempfile
from urllib import request,parse
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json,save_new,sha_file,digest
from barbet_pipeline.full_corpus import STATE
from barbet_pipeline.full_publication import REPO,BRANCH


def scan(receipt):
    counts=Counter();groups=defaultdict(Counter);lengths=Counter();max_total=max_prompt=0
    for item in receipt['files']:
        if '/records/' not in item['remote']:continue
        original=Path(item['local'])
        with tempfile.TemporaryDirectory(dir=STATE,prefix='stats-download-') as temporary:
            path=original
            if not path.exists():
                path=Path(temporary)/original.name
                url=f'https://huggingface.co/datasets/{REPO}/resolve/{BRANCH}/'+parse.quote(item['remote'],safe='/')
                with request.urlopen(url,timeout=180) as source,path.open('xb') as target:
                    while True:
                        chunk=source.read(1024*1024)
                        if not chunk:break
                        target.write(chunk)
            if sha_file(path)!=item['sha256']:raise ValueError('Statistics input bytes changed')
            with path.open('rb') as handle:
                parquet=pq.ParquetFile(handle)
                for batch in parquet.iter_batches(batch_size=8192,columns=['source','domain','split','processing_status','pangolin_tokens','pangolin_supervised_tokens','memory_placement','input_view_changed','natural_answer_rewritten','skill_prefill_updated','inference_decisions_json']):
                    for row in batch.to_pylist():
                        key='|'.join(row[x] for x in ['processing_status','source','domain','split'])
                        groups[key].update({'rows':1,'input_tokens':row['pangolin_tokens'],'supervised_tokens':row['pangolin_supervised_tokens']})
                        counts['rows']+=1;counts['input_tokens']+=row['pangolin_tokens'];counts['supervised_tokens']+=row['pangolin_supervised_tokens']
                        counts['memory:'+row['memory_placement']]+=1
                        for field in ['input_view_changed','natural_answer_rewritten','skill_prefill_updated']:counts[field]+=row[field]
                        bucket=next(n for n in [2048,4096,8192,16384,32768,65536,131072] if row['pangolin_tokens']<=n)
                        lengths[str(bucket)]+=1;max_total=max(max_total,row['pangolin_tokens'])
                        for decision in json.loads(row['inference_decisions_json']):
                            counts['decision_points']+=1
                            max_prompt=max(max_prompt,decision['served_chat_tokens'],decision['served_completion_tokens'])
    assert counts['rows']==receipt['counts']['exported']
    return {'counts':dict(counts),'exposure_groups':{k:dict(v) for k,v in groups.items()},'length_buckets':dict(lengths),
            'maximum_total_pangolin_tokens':max_total,'maximum_served_prompt_tokens':max_prompt}

def main():
    finished=set();parts=[]
    while len(finished)<1504:
        pending=[p for p in sorted((STATE/'receipts').glob('*.json')) if p.name not in finished]
        if not pending:time.sleep(5);continue
        for path in pending:
            result_path=STATE/'statistics'/path.name
            if result_path.exists():result=read_json(result_path)
            else:
                result=scan(read_json(path));result['source_receipt_sha256']=sha_file(path);save_new(result_path,result)
            finished.add(path.name);parts.append(result)
        print(json.dumps({'statistics_shards':len(finished),'rows':sum(p['counts']['rows'] for p in parts)}),flush=True)
    counts=Counter();groups=defaultdict(Counter);lengths=Counter()
    for p in parts:
        counts.update(p['counts']);lengths.update(p['length_buckets'])
        for key,value in p['exposure_groups'].items():groups[key].update(value)
    result={'source_shards':len(parts),'counts':dict(counts),'exposure_groups':{k:dict(v) for k,v in groups.items()},
            'length_buckets':dict(lengths),'maximum_total_pangolin_tokens':max(p['maximum_total_pangolin_tokens'] for p in parts),
            'maximum_served_prompt_tokens':max(p['maximum_served_prompt_tokens'] for p in parts),'all_exported_rows_measured':True}
    save_new(STATE/'statistics-summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='exposure_groups'}))

if __name__=='__main__':main()
