"""All-row portable Chat/Completion budget proof with live tokenizer controls.

Run in the existing Qwen serving environment. Only tokenizer work is performed;
no answer is sampled and no GPU model is instantiated by this script.
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from urllib import request
from collections import Counter
from transformers import AutoTokenizer
# Reuse the unchanged data environment's Arrow wheel without installing packages.
sys.path.append('/root/barbet-context-sft-data/.venv/lib/python3.12/site-packages')
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json,save_new,digest,sha_file

TOKENIZER=None
MODEL='Qwen/Qwen3.8-Flash-Next-FP8'
MODEL_DIR='/root/barbet-context-sft-data/inference/qwen38/model-fp8'


def token_modes(decision):
    text=TOKENIZER.apply_chat_template(decision['chat'],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    chat=TOKENIZER.encode(text,add_special_tokens=False)
    completion=TOKENIZER.encode(decision['completion_prompt'],add_special_tokens=False)
    return chat,completion


def live_control(decision):
    chat,completion=token_modes(decision)
    results=[]
    for mode,ids in [('chat',chat),('completion',completion)]:
        body={'model':MODEL,'add_special_tokens':False}
        if mode=='chat':body.update(messages=decision['chat'],add_generation_prompt=True,
                                    chat_template_kwargs={'enable_thinking':False})
        else:body['prompt']=decision['completion_prompt']
        req=request.Request('http://127.0.0.1:8002/tokenize',data=json.dumps(body).encode(),
                            headers={'Content-Type':'application/json'})
        with request.urlopen(req,timeout=60) as response:actual=json.load(response)
        if actual['tokens']!=ids or actual['count']!=len(ids) or actual['max_model_len']!=32768:
            raise ValueError('Local served-tokenizer path differs from live vLLM')
        results.append({'mode':mode,'token_ids_sha256':digest(ids),'count':len(ids),
                        'live_exact_ids_equal':True,'max_model_len':actual['max_model_len']})
    return results


def scan(item):
    path=item['path']
    if sha_file(path)!=item['sha256']:raise ValueError('Records bytes changed before transport audit')
    count=0;decisions=0;maxchat=maxcompletion=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['inference_decisions_json','record_id']):
        for row in batch.to_pylist():
            count+=1
            for d in json.loads(row['inference_decisions_json']):
                if digest(d['chat'])!=d['chat_sha256'] or digest(d['completion_prompt'])!=d['completion_sha256']:
                    raise ValueError('Answer-free input identity differs')
                chat,completion=token_modes(d)
                if max(len(chat),len(completion))+d['output_reserve_tokens']>32768:
                    raise ValueError('Served-tokenizer input plus 4096 reserve exceeds total budget')
                decisions+=1;maxchat=max(maxchat,len(chat));maxcompletion=max(maxcompletion,len(completion))
    return {'records_path':path,'records_sha256':item['sha256'],'rows':count,'decision_points':decisions,
            'max_chat_tokens':maxchat,'max_completion_tokens':maxcompletion,
            'output_reserve_tokens':4096,'served_total_budget':32768,'truncated':False}


def main(manifest_path,output):
    global TOKENIZER
    TOKENIZER=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False)
    manifest=read_json(manifest_path)
    items=[x for x in manifest['files'] if '/records/' in x['path']]
    first=pq.read_table(items[0]['path'],columns=['inference_decisions_json']).slice(0,1).to_pylist()[0]
    controls=[]
    for d in json.loads(first['inference_decisions_json']):controls+=live_control(d)
    save_new(str(output)+'.live-controls.json',{'controls':controls,'server':'localhost:8002',
        'model':MODEL,'tokenizer_sha256':sha_file(Path(MODEL_DIR)/'tokenizer.json')})
    with ProcessPoolExecutor(4,mp_context=get_context('fork')) as pool:results=list(pool.map(scan,items))
    report={'model':MODEL,'model_revision':'236dfdf285828023ca3bcd3f37366c58a3469b13',
        'tokenizer_sha256':sha_file(Path(MODEL_DIR)/'tokenizer.json'),
        'tokenizer_config_sha256':sha_file(Path(MODEL_DIR)/'tokenizer_config.json'),
        'record_rows':sum(x['rows'] for x in results),'decision_points':sum(x['decision_points'] for x in results),
        'max_chat_tokens':max(x['max_chat_tokens'] for x in results),
        'max_completion_tokens':max(x['max_completion_tokens'] for x in results),
        'output_reserve_tokens':4096,'served_total_budget':32768,'shards':results,
        'all_rows_both_modes_measured':True,'no_answer_generation':True,
        'barbet_vllm_backend_certification':False,'native1M_certification':False}
    save_new(output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='shards'}))


if __name__=='__main__':main(*sys.argv[1:])
