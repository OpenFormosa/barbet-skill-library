"""Implemented CPU data commands and a real native Barbet inference entrypoint.

Training, promotion and Megatron parity are deliberately absent until their
engineering and experimental gates exist. No CLI stub pretends to train.
"""
import argparse
import json
from pathlib import Path
from .common import read_json,save_new


def main(argv=None):
    p=argparse.ArgumentParser(prog='python -m barbet_pipeline')
    sub=p.add_subparsers(dest='command',required=True)
    audit=sub.add_parser('audit',help='Reindex every admitted source row without changing source bytes')
    audit.add_argument('--work',required=True);audit.add_argument('--output',required=True)
    audit.add_argument('--workers',type=int,default=4)
    data=sub.add_parser('build-data',help='Compile natural source views or the reviewed semantic pilot')
    data.add_argument('--kind',choices=['reading','semantic'],required=True)
    data.add_argument('--work');data.add_argument('--wording');data.add_argument('--private-output')
    data.add_argument('--output',required=True);data.add_argument('--tokenizer',required=True)
    validate=sub.add_parser('validate-data',help='Rebuild exact text, token IDs, labels and causal prefixes')
    validate.add_argument('--manifest',required=True);validate.add_argument('--tokenizer',required=True)
    validate.add_argument('--output',required=True)
    run=sub.add_parser('run-barbet',help='Hash-bound greedy native Barbet with the public tool harness')
    run.add_argument('--model-folder',required=True);run.add_argument('--model-manifest',required=True)
    run.add_argument('--episode',required=True);run.add_argument('--environment',required=True)
    run.add_argument('--output',required=True);run.add_argument('--max-steps',type=int,default=6)
    run.add_argument('--output-tokens',type=int,default=512);run.add_argument('--total-tokens',type=int,default=8192)
    a=p.parse_args(argv)
    if a.command=='audit':
        from .corpus import audit_corpus
        r=audit_corpus(a.work,a.output,workers=a.workers)
        print(json.dumps(r['counts']));return
    if a.command=='build-data':
        if a.kind=='reading':
            if not a.work:p.error('--kind reading requires --work')
            from .reading import reprocess_reading
            r=reprocess_reading(a.work,a.output,a.tokenizer)
        else:
            if not a.wording or not a.private_output:p.error('--kind semantic requires --wording and --private-output')
            from .pilot import compile_pilot
            r=compile_pilot(a.wording,a.output,a.private_output,a.tokenizer)
        print(json.dumps(r['counts']));return
    if a.command=='validate-data':
        from collections import Counter
        from .export import load_tokenizer,validate_shard
        tokenizer=load_tokenizer(a.tokenizer);manifest=read_json(a.manifest)
        totals=Counter();pairs=[]
        for f in manifest['files']:
            path=f['path']
            if '/records/' not in path:continue
            result=validate_shard(path,path.replace('/records/','/tokenized/'),tokenizer)
            totals.update(result);pairs.append({'records':path,'checks':result})
        save_new(a.output,{'counts':dict(totals),'shards':pairs,'exact_renderer_and_unshifted_labels':True,
                          'private_future_visibility_checked':True,'model_learning':'NOT_RUN'})
        print(json.dumps(dict(totals)));return
    from .executors import Executor,StateLedger
    from .harness import run_public_episode
    from barbet_inference.barbet_backend import NativeBarbetBackend
    env=read_json(a.environment)
    if set(env)-{'tables','evidence','skills','state_ledger','scope'}:raise ValueError('Public environment allowlist violation')
    ledger=StateLedger();saved=env.get('state_ledger',{'revision':0,'operations':{}})
    ledger.revision=saved['revision'];ledger.operations=saved['operations']
    executor=Executor(tables=env.get('tables'),evidence=env.get('evidence'),skills=env.get('skills'),ledger=ledger)
    backend=NativeBarbetBackend(a.model_folder,read_json(a.model_manifest),max_model_len=a.total_tokens)
    result=run_public_episode(backend,read_json(a.episode),executor,max_steps=a.max_steps,output_tokens=a.output_tokens)
    save_new(a.output,{'backend':backend.metadata,**result})
    print(json.dumps({'status':result['status'],'output':a.output}))


if __name__=='__main__':main()
