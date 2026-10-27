"""Prepare, atomically publish, and anonymously reread the new-only HF release.

No source code execution or model training. One explicit publisher, pinned parent,
immutable per-wave data, complete remote SHA256 verification. The old v1 release
stays available through Hub history; it is absent from active v2 configs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote

import requests
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
from scripts.build_memory_sft_v2 import WORK, STATE, RAW, REPO, file_sha, save
from scripts.package_memory_sft_v2 import PUBLIC, WAVE

DATASET='OpenFormosa/barbet-long-context-sft'
OLD='692018a4ad267e6bfbcdac624e7e1f7d3b2fba3d'


def safe_copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and file_sha(source)!=file_sha(dest):
        raise ValueError('immutable metadata conflict: '+str(dest))
    if not dest.exists():shutil.copyfile(source,dest)


def prepare():
    summary=json.loads((STATE/'public-package-summary.json').read_text())
    c=summary['counts'];n=c['accepted'];u=summary['unique_counts']
    if n!=u['record_id'] or n<=0:raise ValueError('invalid accepted identity count')
    for key in ['all_records_and_arrays_reread','all_answer_free_qwen_chat_and_completion_budgets_measured',
                'all_training_and_native_minicpm_budgets_with_4096_reserve']:
        if summary.get(key) is not True:raise ValueError('missing publication gate '+key)
    for f in summary['files']:
        p=PUBLIC/f['path']
        if p.stat().st_size!=f['bytes'] or file_sha(p)!=f['sha256']:
            raise ValueError('changed verified bytes')
    reports=PUBLIC/'reports'/WAVE;reports.mkdir(parents=True,exist_ok=True)
    # Only reviewed aggregate information; no private prompts, source oracles,
    # raw semantic reviews, repair payloads or credential-bearing environment.
    save(reports/'dataset-summary.json',{**{k:v for k,v in summary.items() if k!='files'},
                                       'status':'LOCALLY_VERIFIED_AT_PACKAGING; see PUBLICATION.json for remote status'})
    save(reports/'data-files.json',summary['files'])
    for name in ['publication-quality-policy-v1.json','official-template-proof.json',
                 'served-tokenizer-proof.json','v26-data-design-response.json']:
        value=json.loads((STATE/name).read_text())
        # Proofs are recursively stripped of local filesystem locations.
        def sanitize(v):
            if isinstance(v,dict):return {k:sanitize(x) for k,x in v.items() if k not in {'path','local','request','response','prompt','packet'}}
            if isinstance(v,list):return [sanitize(x) for x in v]
            if isinstance(v,str) and v.startswith('/root/'):return '[local evidence retained]'
            return v
        save(reports/name,sanitize(value))
    audit=json.loads((STATE/'semantic-audit-v1/summary.json').read_text())
    expected_controls={'control-wrong-number':'REJECT','control-unsupported-success':'REJECT',
                       'control-positive-number':'ACCEPT','control-positive-evidence':'ACCEPT'}
    actual_controls={r['id']:r['verdict'] for r in audit['controls']}
    if actual_controls!=expected_controls:raise ValueError('semantic audit control failure')
    save(reports/'semantic-audit-summary.json',{
        'scope':'40 stratified pilot samples; not a representative population accuracy estimate',
        'verdicts':dict(Counter(r['verdict'] for r in audit['results'])),
        'first_nonaccept_record_ids':audit['excluded_record_ids'],
        'controls':actual_controls,'first_holds_preserved':True,
        'all_rows_semantically_verified':False})
    source_lock=json.loads((STATE/'build-source-lock.json').read_text())
    source_files=[{k:v for k,v in f.items() if k!='local'} for f in source_lock['files']]
    save(PUBLIC/'provenance'/WAVE/'source-files.json',source_files)
    for name in ['memory_sft_v2.json','memory_sft_v2_scale_10m.json']:
        safe_copy(REPO/'configs'/name,PUBLIC/'recipe'/name)
    names=['contract-tests.log','context-agent-tests.log','dataset-adapter-tests.log','existing-inference-tests.log','release-guard-tests.log']
    for name in names:safe_copy(STATE/name,reports/name)
    source_repos={f['repo']:f['revision'] for f in source_files}
    for entry in json.loads((RAW/'upstream-notices/lock.json').read_text()):
        if entry.get('repo'):source_repos.setdefault(entry['repo'],entry.get('revision','main'))
    for sid in source_repos:
        card=RAW/'upstream-notices'/(sid.replace('/','--')+'.README.md')
        if card.exists():safe_copy(card,PUBLIC/'provenance'/WAVE/'upstream-notices'/card.name)
    notice=['# Source attribution and changes','',
        'This is a mixed-license collection, not a relicensing of upstream content. Each records row retains its source, upstream source and license. Preserve applicable attribution and share-alike conditions when redistributing. The library code license does not replace the data licenses.','',
        'Changes: normalized causal messages and typed tools; removed private reasoning fields; added applicable skills; represented source passages as identified memory; constructed real deterministic retrieval observations where supported; deduplicated and assigned source-family splits; rendered and tokenized two training formats. Imported targets are not claimed to have been independently verified for correctness.','',
        'Source authors and snapshots:','']
    notice += [f'- [{sid}](https://huggingface.co/datasets/{sid}/tree/{rev}) — revision `{rev}`.' for sid,rev in sorted(source_repos.items())]
    notice+=['','The active data include MIT, Apache-2.0, CC-BY-4.0 and CC-BY-SA-4.0 subsets; filter the per-row license before use. Held or reference-only sources are not thereby redistributed. No UltraData-SFT-Agent artifacts are rehosted.']
    (PUBLIC/'NOTICE.md').write_text('\n'.join(notice)+'\n')
    yaml=['---','pretty_name: Barbet Memory, Skills and Tool SFT v2','license: other','license_name: mixed-upstream-licenses','license_link: NOTICE.md','task_categories:','- text-generation','language:','- en','- zh','tags:','- retrieval-augmented-generation','- tool-use','- agent','- long-context','configs:']
    for config in ['records','tokenized','minicpm5_tokenized']:
        yaml += [f'- config_name: {config}']
        if config=='records':yaml += ['  default: true']
        yaml += ['  data_files:','  - split: train',f'    path: data/wave-*/{config}/train-*.parquet','  - split: validation',f'    path: data/wave-*/{config}/validation-*.parquet']
    yaml+=['---','']
    md=f'''# Barbet Memory, Skills and Tool SFT v2

The active release contains **{n:,} new, deduplicated episodes**, **{u['query_hash']:,} distinct normalized first-user queries**, and **{u['family_id']:,} conservative split families**. Train: **{c['split:train']:,}**; validation: **{c['split:validation']:,}**. Multiple genuinely different trajectories can share a query; these figures are not interchangeable.

The project target is **10,000,000 accepted new episodes**. This is wave 1 ({100*n/10000000:.2f}%); the target is **not yet complete**. Legacy active rows: **0**. The previous 8,159-row release remains reproducible at [its historical commit](https://huggingface.co/datasets/{DATASET}/tree/{OLD}).

## Training views and measured scale

| View | Physical tokens | Assistant-supervised tokens |
|---|---:|---:|
| Pangolin | {c['pangolin_tokens']:,} | {c['pangolin_supervised_tokens']:,} |
| MiniCPM5-2B | {c['minicpm_tokens']:,} | {c['minicpm_supervised_tokens']:,} |

`records` contains full texts, canonical messages, tool schemas, source documents, provenance, target spans and exact token counts. `tokenized` contains Pangolin `input_ids` and unshifted assistant-only `labels`; `minicpm5_tokenized` contains the equivalent native MiniCPM5-2B view. Masked positions have label `-100`; causal trainers shift logits/labels once. Do not concatenate these views as additional independent examples.

All trajectories reserve 4,096 output tokens within a 131,072-token budget under **both** training tokenizers. Text is never padded with unrelated background or truncated to claim compatibility. The curriculum buckets are actual lengths, not a claim that every example is long. There are {c['assistant_turns']:,} supervised assistant turns, {c['tool_observations']:,} tool observations, and {c['actual_retrieval']:,} examples with locally performed deterministic retrieval. The field `recorded_error_followup` flags {c['recorded_error_followup']:,} examples using an error-word heuristic over tool observations; despite its historical name it does **not** establish a real failure, a subsequent repair action or successful refinement.

## Use

```python
from datasets import load_dataset
rows = load_dataset("{DATASET}", "records", split="train", streaming=True)
row = next(iter(rows))
```

Use `barbet_inference.prepare_dataset_request` from the linked library for answer-free Chat or exact-renderer Completion requests. It removes the selected target and all later turns while retaining genuine earlier actions and observations. Tools use an explicit JSON envelope in native chat content; do not add a second tool formatter. Stored training token IDs are specific to their named tokenizer and are not portable inference payloads.

The [skill and harness library](https://github.com/OpenFormosa/barbet-skill-library/tree/feat/memory-harness-sft-v2) provides skill retrieval, causal external memory, typed registered tool callbacks, and bounded refinement using actual verifier feedback. It never executes source programs merely because they appear in a dataset.

## Validation and limits

Every row passed schema/chronology checks, declared-tool and argument validation, assistant-mask checks, exact local Parquet readback, two training-tokenizer budgets, and measured answer-free Chat plus Completion budgets for the pinned served Qwen tokenizer. Global record IDs are unique; normalized first-user query and conservative source-family identities do not cross train/validation. Known local benchmark exact/13-word matches and previously held source jobs are excluded; this is not universal benchmark decontamination.

A first-pass local-Qwen review of **40 stratified pilot examples** produced **34 ACCEPT, 5 REVISE, 1 REJECT**. All non-accept examples and same-query variants remain held. After three problematic search examples, unreviewed search examples are held in this wave. This diagnostic sample is not a population accuracy estimate. Imported answers retain their upstream verification tier; format validation is not proof that every answer, tool choice, code solution or web fact is correct. Recorded upstream tool observations are not claimed to have been rerun locally.

MiniCPM5-2B is the actual 2.517B-parameter, 128K-context reference. Its public agent SFT release reports 483,661 trajectories, not necessarily unique tasks. The >15M core SFT release is a 1B reference and is not a verified exact 2B training mixture. The reference licenses and redistribution conditions are respected; those restricted artifacts are not included here.

No model was trained in this build; no gain over Qwen3.5-9B or MiniCPM5 is established. Data-format compatibility and served-Qwen budget checks do not certify a Barbet Attention/Mamba vLLM backend. Most imported tasks are English; Traditional Chinese skills do not make the dataset predominantly Traditional Chinese.

Read [source attribution](NOTICE.md), [exact counts and limits](reports/{WAVE}/dataset-summary.json), and the [Traditional Chinese report](reports/{WAVE}/report.html). Publication readback status is recorded separately in `PUBLICATION.json` when verified.
'''
    (PUBLIC/'README.md').write_text('\n'.join(yaml)+md)
    # Self-contained HTML: no network assets, no source text or hidden oracles.
    pairs=[('新軌跡',n),('獨立正規化首題',u['query_hash']),('保守題目家族',u['family_id']),('Pangolin 有效 supervised tokens',c['pangolin_supervised_tokens']),('MiniCPM5 有效 supervised tokens',c['minicpm_supervised_tokens'])]
    def table(prefix):
        rows=[(k[len(prefix):],v) for k,v in sorted(c.items()) if k.startswith(prefix)]
        return '<table><tr><th>分類</th><th>筆數</th></tr>'+''.join(f'<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>' for k,v in rows)+'</table>'
    body=f'''<h1>Barbet 外部記憶、技能與工具訓練資料：第一批報告</h1>
<p>目標：一千萬筆通過檢查的新軌跡。目前 {n:,} 筆，進度 {100*n/10000000:.2f}%；尚未達標，goal 持續進行。舊版 8,159 筆已退出現役設定，歷史版本保留。</p>
<p>本次改造把原始文件、工具規格與已發生的觀察放入答案前可見的上下文；以既有技能程序協助讀證據、選工具、保留參數型別及根據真實回饋修正。沒有以無關長背景、同題前後位置變體或不同 tokenizer 視圖灌水。</p>
<ul>{''.join(f'<li>{html.escape(k)}：{v:,}</li>' for k,v in pairs)}</ul>
<h2>v26 的問題如何反映在資料設計</h2><p>既有診斷中，修正原生格式後，只有破折號的輸出由 257 次降到 0，但數學仍為 0/127、工具完整正確為 4/127。工具可解析有 108 筆；其中 43 筆選對名稱、39 筆仍有參數錯誤。這表示格式之外，基本計算、工具選擇及參數綁定都需要資料支持。此處加入有來源的數學步驟、完整工具規格、精確型別驗證與先呼叫後觀察的軌跡；沒有把資料建置成功當成能力改善的實驗結論。</p>
<h2>領域組成</h2>{table('domain:')}<h2>實際長度區間上限</h2>{table('length_bucket:')}
<p>長度以兩套訓練 tokenizer 的較大值分桶，另保留 4,096 token。短題維持短題；64K／128K 課程的不足必須靠有用文件、真實多步任務補充。</p>
<h2>驗證層級</h2>{table('verification_tier:')}<h2>第一批剔除原因</h2>{table('excluded:')}
<p>40 筆分層診斷抽查為 34 ACCEPT、5 REVISE、1 REJECT；保留首次失敗且不重抽求過關。搜尋子集出現證據鏈不足，因此此批只留下抽查首次通過的搜尋例。這個樣本不能估計全資料正確率，其餘匯入解答仍保留上游驗證層級。所有資料均經工程檢查，不等於所有語意都正確。</p>
<h2>Harness 與推論</h2><p>新增可檢索的外部記憶、技能檢索、具 JSON Schema 的註冊工具，以及最多兩次修正的真實 verifier 回饋。工具回傳只能在實際 callback 之後出現；驗證必須綁定當次答案雜湊；反覆相同答案會停止。未提供 callback 不會捏造執行結果。新增工具格式有明確的資料轉推論介面。</p>
<h2>MiniCPM5 參考與後續擴充</h2><p>官方 MiniCPM5-2B 實際約 25.17 億參數、128K 上下文；agent SFT 公開的是 483,661 條軌跡。超過 1,500 萬筆的 core SFT 屬於 1B 參考，不應冒稱已核實的 2B 完整配比。本專案用相同原生 tokenizer 衡量有效監督量，不下載模型權重訓練。</p>
<p>後續依序處理 OpenOrca、Toucan 與可追溯的新來源，沿用來源家族切分、舊失敗隔離、逐筆型別與 token 檢查。不同教師對同題的回答與不同工具結果會另外揭露，不當成同等數量的獨立難題。資料仍以英文為主，繁體中文技能前綴不等於完成台灣語料在地化。</p>
<p>目前未訓練模型，尚未證明 test-time 能力提升或比擬 9B。資料格式與 Qwen 實際 tokenizer 預算通過，也不代表 Barbet 混合模型後端已受 vLLM 支援。</p>'''
    (reports/'report.html').write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Barbet v2 第一批資料報告</title><style>body{font:17px/1.8 system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#193042}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{text-align:left;padding:8px;border-bottom:1px solid #ccd}h1,h2{line-height:1.4}</style>'+body+'</html>')
    manifest=[{'path':str(p.relative_to(PUBLIC)),'bytes':p.stat().st_size,'sha256':file_sha(p)} for p in sorted(PUBLIC.rglob('*')) if p.is_file() and p.name!='PUBLICATION.json']
    save(STATE/'publication-manifest-v1.json',manifest)
    save(STATE/'GOAL_PROGRESS.json',{'target':10000000,'locally_verified_new_episodes':n,'published_new_episodes':0,'phase':'ready_to_publish','goal_complete':False,'wave':WAVE})
    print(json.dumps({'files':len(manifest),'bytes':sum(f['bytes'] for f in manifest),'accepted':n}),flush=True)


def publish():
    manifest=json.loads((STATE/'publication-manifest-v1.json').read_text())
    api=HfApi();receipt=STATE/'publication-data-commit.json'
    if receipt.exists():raise ValueError('already committed; use verify')
    # All bytes were rehashed by prepare. Recheck immediately before upload.
    for f in manifest:
        if file_sha(PUBLIC/f['path'])!=f['sha256']:raise ValueError('manifest changed')
    head=api.repo_info(DATASET,repo_type='dataset').sha
    if head!=OLD:raise ValueError('remote head changed; reconcile before publication: '+head)
    existing=api.list_repo_files(DATASET,repo_type='dataset',revision=head)
    prefixes=sorted({p.split('/')[0] for p in existing if '/' in p})
    operations=[CommitOperationDelete(p,is_folder=True) for p in prefixes]
    oldfiles=[p for p in existing if '/' not in p and p not in {'.gitattributes','README.md'}]
    operations += [CommitOperationDelete(p,is_folder=False) for p in oldfiles]
    operations += [CommitOperationAdd(f['path'],PUBLIC/f['path']) for f in manifest]
    result=api.create_commit(DATASET,repo_type='dataset',operations=operations,
        commit_message='Publish new-only memory, skills and tool SFT v2 wave 1',
        commit_description='Replace active legacy configs with verified new v2 data; retain history. 10M target remains in progress. No model training or efficacy claim.',
        parent_commit=head,num_threads=4)
    save(receipt,{'dataset':DATASET,'parent':head,'commit':result.oid,
                  'manifest_sha256':file_sha(STATE/'publication-manifest-v1.json'),
                  'files':len(manifest),'deleted_active_prefixes':prefixes,'old_history_retained':True})
    print(json.dumps({'commit':result.oid,'files':len(manifest)}),flush=True)


def anonymous_file(item,revision):
    url=f'https://huggingface.co/datasets/{DATASET}/resolve/{revision}/'+quote(item['path'],safe='/')
    for attempt in range(4):
        try:
            h=hashlib.sha256();count=0
            # Direct public HTTP; no HF token/session authorization headers.
            with requests.Session() as session:
                session.trust_env=False  # no netrc credentials or inherited proxy auth
                with session.get(url,stream=True,timeout=(30,180)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(4*1024**2):h.update(chunk);count+=len(chunk)
            if count!=item['bytes'] or h.hexdigest()!=item['sha256']:raise ValueError('public bytes differ '+item['path'])
            return {'path':item['path'],'sha256':h.hexdigest(),'bytes':count,'anonymous':True}
        except (requests.RequestException,ValueError):
            if attempt==3:raise
            time.sleep(min(2**attempt,8))


def verify():
    receipt=json.loads((STATE/'publication-data-commit.json').read_text())
    manifest=json.loads((STATE/'publication-manifest-v1.json').read_text())
    checks=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,result in enumerate(pool.map(lambda f:anonymous_file(f,receipt['commit']),manifest),1):
            checks.append(result)
            if n%100==0:print(json.dumps({'verified_files':n,'total':len(manifest)}),flush=True)
    # Anonymous pinned tree checks ensure no old active payload survived.
    api=HfApi(token=False)
    paths=set(api.list_repo_files(DATASET,repo_type='dataset',revision=receipt['commit']))
    expected={f['path'] for f in manifest}|{'.gitattributes'}
    if paths!=expected:raise ValueError('unexpected/missing public paths')
    save(STATE/'anonymous-readback-v1.json',{'commit':receipt['commit'],'checks':checks,'all_files_verified':True})
    summary=json.loads((STATE/'public-package-summary.json').read_text())
    public={'dataset':DATASET,'data_commit':receipt['commit'],'previous_release':OLD,
        'anonymous_full_file_sha256_readback':True,'verified_files':len(checks),
        'verified_bytes':sum(c['bytes'] for c in checks),'active_new_episodes':summary['counts']['accepted'],
        'active_legacy_rows':0,'target_accepted_episodes':10000000,'target_complete':False,
        'manifest_sha256':receipt['manifest_sha256'],'readback_receipt_sha256':file_sha(STATE/'anonymous-readback-v1.json')}
    save(PUBLIC/'PUBLICATION.json',public)
    writer=HfApi()
    if writer.repo_info(DATASET,repo_type='dataset').sha!=receipt['commit']:raise ValueError('head moved during verification')
    result=writer.create_commit(DATASET,repo_type='dataset',
        operations=[CommitOperationAdd('PUBLICATION.json',PUBLIC/'PUBLICATION.json')],
        commit_message='Record anonymous full-file readback for v2 wave 1',parent_commit=receipt['commit'])
    anonymous_file({'path':'PUBLICATION.json','bytes':(PUBLIC/'PUBLICATION.json').stat().st_size,
                    'sha256':file_sha(PUBLIC/'PUBLICATION.json')},result.oid)
    save(STATE/'publication-final-commit.json',{**public,'final_head':result.oid,'publication_receipt_anonymous_verified':True})
    save(STATE/'GOAL_PROGRESS.json',{'target':10000000,'locally_verified_new_episodes':summary['counts']['accepted'],
        'published_new_episodes':summary['counts']['accepted'],'phase':'wave_1_published_continuing_expansion',
        'goal_complete':False,'wave':WAVE,'final_head':result.oid})
    print(json.dumps({'head':result.oid,'accepted':summary['counts']['accepted'],'target_complete':False}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','publish','verify']);args=parser.parse_args()
    # Cross-process publisher serialization; prepare is local and does not mutate HF.
    with (STATE/'publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        globals()[args.phase]()
