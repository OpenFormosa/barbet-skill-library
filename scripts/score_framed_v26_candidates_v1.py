"""Same frozen numerical scorer with an explicitly bound lossless frame reader.

This prospective runner changes candidate storage only. It never edits original
JSONL cohort code, numerical calibration, model weights or first dispositions.
"""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import sqlite3
import time
import torch
from scripts.calibrate_v26_answer_ppl import ROOT,STATE,load,sha_file
from barbet_inference.candidate_archive import load_candidate, FORMAT
from barbet_inference.answer_selection import score_batch,select_minimum_ppl
from barbet_inference.memory import canonical
import hashlib


def main(bucket,limit,label):
    if not 0<=bucket<8 or limit<0:raise ValueError('bucket or limit bound')
    cohort=json.loads((STATE/'candidate-policy-binding.json').read_text())
    for key,path in [('native_runner_sha256',Path(__file__)),
                     ('storage_loader_sha256',ROOT/'barbet_inference/candidate_archive.py'),
                     ('canonicalizer_sha256',ROOT/'scripts/prepare_v26_answer_candidates.py')]:
        if cohort[key]!=sha_file(path):raise ValueError('prospective cohort binding differs: '+key)
    identity=json.loads((STATE/'episode-identity-v1/summary.json').read_text())
    if identity['status']!='COMPLETE' or not identity['all_original_candidates_covered']:
        raise ValueError('complete episode identity coverage required')
    validation=json.loads((STATE/'canonical-single-v3-validation.json').read_text())
    policy_sha=sha_file(ROOT/'configs/memory_sft_v2_v26_selection.json')
    if (not validation['all_checks_passed'] or validation['policy_sha256']!=policy_sha
            or validation['scorer_sha256']!=sha_file(ROOT/'barbet_inference/answer_selection.py')):
        raise ValueError('missing or stale canonical score validation')
    summary=json.loads((STATE/'candidate-index-summary.json').read_text())
    if summary['status']!='INDEXED_NOT_YET_PPL_SELECTED':raise ValueError('candidate index not frozen')
    files=json.loads((STATE/'candidate-input-files.json').read_text())
    for spec in files:
        if spec.get('format')!=FORMAT or sha_file(Path(spec['path']))!=spec['sha256']:
            raise ValueError('candidate archive format or bytes differ')
    folder=STATE/'scores'/label;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/f'bucket-{bucket}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bindings={'policy_sha256':policy_sha,'candidate_manifest_sha256':sha_file(STATE/'candidate-input-files.json'),
        'candidate_summary_sha256':sha_file(STATE/'candidate-index-summary.json'),'bucket':bucket,'limit':limit,
        'scorer_sha256':sha_file(ROOT/'barbet_inference/answer_selection.py'),
        'runner_sha256':sha_file(Path(__file__)),'canonicalizer_sha256':sha_file(ROOT/'scripts/prepare_v26_answer_candidates.py'),
        'storage_loader_sha256':sha_file(ROOT/'barbet_inference/candidate_archive.py'),
        'storage_format':FORMAT,'cohort_binding_sha256':sha_file(STATE/'candidate-policy-binding.json'),
        'canonical_batch_size':1,'model_revision':'67131067ff1825179d5e38e91d499159669e9c66'}
    protocol=hashlib.sha256(canonical({k:v for k,v in bindings.items() if k not in ['bucket','limit']}).encode()).hexdigest()
    binding_path=folder/f'bucket-{bucket}.binding.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text())!=bindings:raise ValueError('frozen scorer binding changed')
    else:binding_path.write_text(json.dumps(bindings,indent=2)+'\n')
    model,renderer,_=load()
    source=sqlite3.connect(f'file:{STATE / "candidates.sqlite"}?mode=ro',uri=True);source.row_factory=sqlite3.Row
    output=sqlite3.connect(folder/f'bucket-{bucket}.sqlite')
    output.execute('CREATE TABLE IF NOT EXISTS scores(candidate_identity TEXT PRIMARY KEY,input_hash TEXT,record_id TEXT,score_json TEXT)')
    output.execute('CREATE TABLE IF NOT EXISTS decisions(input_hash TEXT PRIMARY KEY,status TEXT,winner TEXT,report_json TEXT)')
    groups=list(source.execute('SELECT input_hash,candidates FROM groups WHERE bucket=? AND candidates>1 ORDER BY input_hash'+(' LIMIT ?' if limit else ''),(bucket,limit) if limit else (bucket,)))
    counts=Counter();start=time.monotonic()
    for n,g in enumerate(groups,1):
        prior=output.execute('SELECT status FROM decisions WHERE input_hash=?',(g['input_hash'],)).fetchone()
        if prior:counts['resumed:'+prior[0]]+=1;continue
        pointers=list(source.execute('SELECT * FROM candidates WHERE input_hash=? ORDER BY record_id',(g['input_hash'],)))
        scored=[];error=None
        for pointer in pointers:
            existing=output.execute('SELECT score_json FROM scores WHERE candidate_identity=?',(pointer['candidate_identity'],)).fetchone()
            if existing:result=json.loads(existing[0]);scored.append(result);continue
            try:
                row=load_candidate(pointer,files);item=renderer.render(row['messages'],row['tools'])
                if item['input_hash']!=g['input_hash']:raise ValueError('rendered input identity differs')
                result=score_batch(model,[item])[0]
                result.update(record_id=row['record_id'],eligible=True,protocol_sha256=protocol,
                              candidate_identity=pointer['candidate_identity'])
                output.execute('INSERT INTO scores VALUES (?,?,?,?)',(pointer['candidate_identity'],g['input_hash'],row['record_id'],canonical(result)));output.commit()
                scored.append(result);counts['answers_scored']+=1;counts['sequence_tokens']+=result['sequence_tokens'];counts['answer_tokens']+=result['answer_tokens']
            except (ValueError,RuntimeError) as exc:
                error={'type':type(exc).__name__,'message':str(exc)[:500],'candidate_identity':pointer['candidate_identity']}
                if isinstance(exc,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
                break
        if error is None and len(scored)==g['candidates']:
            best=select_minimum_ppl(scored)
            report={'input_hash':g['input_hash'],'candidates':len(scored),'winner':best['candidate_identity'],
                    'winner_record_id':best['record_id'],'minimum_ppl':best['ppl'],'mean_nll':best['mean_nll'],
                    'protocol_sha256':protocol,'status':'selected','not_a_correctness_proof':True}
        else:report={'input_hash':g['input_hash'],'status':'held','candidates':g['candidates'],'scored':len(scored),'error':error,'winner':None,'protocol_sha256':protocol}
        output.execute('INSERT INTO decisions VALUES (?,?,?,?)',(g['input_hash'],report['status'],report['winner'],canonical(report)));output.commit()
        counts[report['status']]+=1
        if n%100==0 or n==len(groups):
            print(json.dumps({'bucket':bucket,'processed_groups':n,'total':len(groups),'counts':dict(counts),'seconds':time.monotonic()-start}),flush=True)
    total={r[0]:r[1] for r in output.execute('SELECT status,COUNT(*) FROM decisions GROUP BY status')}
    result={'bucket':bucket,'groups_requested':len(groups),'counts_this_process':dict(counts),'total_decisions':total,
        'seconds_this_process':time.monotonic()-start,'protocol_sha256':protocol,'model_weights_trained':False,
        'status':'BUCKET_COMPLETE','limited_pilot':bool(limit),'full_goal_complete':False}
    output.close();source.close();(folder/f'bucket-{bucket}.summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--bucket',type=int,required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--label',default='canonical-v2');args=p.parse_args()
    if not args.label.replace('-','').isalnum():raise ValueError('invalid output label')
    STATE=args.state.resolve()
    main(args.bucket,args.limit,args.label)
