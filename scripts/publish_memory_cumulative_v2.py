"""Publish cumulative v2 waves after exact compaction and all budget checks."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import html
import json
import shutil
import subprocess
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
from scripts.compact_memory_release_v2 import DEST, STATE, FIRST, SECOND
from scripts.build_memory_sft_v2 import RAW, REPO, file_sha, save
from scripts.verify_memory_publication_v2 import verify_file

DATASET='OpenFormosa/barbet-long-context-sft'


def prepare():
    if (STATE/'SUPERSEDED.json').exists():raise ValueError('superseded by user v26 minimum-PPL selection')
    if (STATE/'publication-data-commit.json').exists():raise ValueError('cannot repackage committed data')
    summary=json.loads((STATE/'summary.json').read_text());c=summary['counts'];u=summary['unique_counts']
    if not summary['all_retained_records_and_token_arrays_equal'] or not summary['all_budget_proofs_preserved']:raise ValueError('missing compaction gates')
    for f in summary['files']:
        if file_sha(DEST/f['path'])!=f['sha256']:raise ValueError('compact bytes changed')
    parent=json.loads((FIRST/'publication-final-commit.json').read_text())['final_head']
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    reports=DEST/'reports/cumulative-0002';reports.mkdir(parents=True,exist_ok=True)
    save(reports/'summary.json',{**{k:v for k,v in summary.items() if k!='files'},'status':'LOCALLY_VERIFIED_AT_PACKAGING','library_commit':code})
    save(reports/'data-files.json',summary['files'])
    save(reports/'quality-change.json',{'new_declared_argument_exclusions':c.get('excluded:new_declared_argument_guard',0),
        'duplicate_openorca_queries_excluded':c.get('excluded:duplicate_openorca_query',0),
        'rule':'Hold unconfirmed root function arguments when a schema enumerates parameters but defaults to permissive additionalProperties; preserve explicitly open interfaces.',
        'not_a_semantic_accuracy_estimate':True,'old_release_retained_by_commit':parent})
    source_files=[];audits={}
    for name,state in [('wave-0001',FIRST),('wave-0002',SECOND)]:
        source_files += [{k:v for k,v in f.items() if k!='local'} for f in json.loads((state/'build-source-lock.json').read_text())['files']]
        audit=json.loads((state/'semantic-audit-v1/summary.json').read_text())
        audits[name]={'samples':len(audit['results']),'verdicts':audit['counts'],
            'first_nonaccept_ids':audit['excluded_record_ids'],'all_rows_semantically_verified':False,
            'controls':[{'id':r['id'],'verdict':r['verdict']} for r in audit['controls']]}
    save(reports/'semantic-audit-scopes.json',audits)
    save(DEST/'provenance/source-files.json',source_files)
    for p in (REPO/'configs').glob('memory_sft_v2*.json'):
        dest=DEST/'recipe'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
    save(DEST/'recipe/code.json',{'repo':'OpenFormosa/barbet-skill-library','commit':code,
        'pull_request':'https://github.com/OpenFormosa/barbet-skill-library/pull/6','model_weights_trained':False})
    for name in ['compaction-query-dedup-tests-v2.log','cumulative-publisher-tests.log','public-readback-v2-tests.log','tool-contract-v2-tests.log',
                 'wave2-adapter-tests.log','wave2-split-v2-tests.log','context-agent-v2-tests.log',
                 'contract-tests.log','dataset-adapter-tests.log','existing-inference-tests.log']:
        shutil.copyfile(FIRST/name,reports/name)
    repos={f['repo']:f['revision'] for f in source_files}
    # Retain the complete earlier upstream attribution, plus the new sources.
    from scripts.package_memory_sft_v2 import PUBLIC as first_public
    text=(first_public/'NOTICE.md').read_text()
    text+='\nAdditional sources in wave 2:\n\n'
    for sid in ['Open-Orca/OpenOrca','Agent-Ark/Toucan-1.5M']:
        text+=f'- [{sid}](https://huggingface.co/datasets/{sid}/tree/{repos[sid]}) — `{repos[sid]}`.\n'
        p=RAW/'scale10m-reference'/(sid.replace('/','--')+'.README.md')
        dest=DEST/'provenance/upstream-notices'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    (DEST/'NOTICE.md').write_text(text)
    header=['---','pretty_name: Barbet Memory, Skills and Tool SFT v2','license: other',
        'license_name: mixed-upstream-licenses',f'license_link: https://huggingface.co/datasets/{DATASET}/blob/main/NOTICE.md',
        'language:','- en','- zh','task_categories:','- text-generation','tags:','- tool-use','- retrieval-augmented-generation','- long-context','configs:']
    for config in ['records','tokenized','minicpm5_tokenized']:
        header += [f'- config_name: {config}']
        if config=='records':header += ['  default: true']
        header += ['  data_files:','  - split: train',f'    path: data/wave-*/{config}/train-*.parquet','  - split: validation',f'    path: data/wave-*/{config}/validation-*.parquet']
    header+=['---','']
    md=f'''# Barbet Memory, Skills and Tool SFT v2

The active release contains **{c['accepted']:,} accepted new episodes**, **{u['query_hash']:,} distinct normalized first-user queries**, and **{u['family_id']:,} conservative split families**. These counts are different measures of scale. Train: {c['split:train']:,}; validation: {c['split:validation']:,}. Legacy active rows: **0**.

The target is **10,000,000 accepted new episodes**; progress is **{c['accepted']/100000:.2f}%**, and the goal remains active. This cumulative release combines two verified waves, removes **{c['excluded:new_declared_argument_guard']:,}** wave-1 rows flagged by the new declared-function-parameter guard, and excludes **{c.get('excluded:duplicate_openorca_query',0):,}** OpenOrca answer variants sharing the same normalized query despite distinct source seed IDs. One OpenOrca response per query remains. Earlier payloads remain available through Hub commit history.

| Training view | Physical tokens | Assistant-supervised tokens |
|---|---:|---:|
| Pangolin | {c['pangolin_tokens']:,} | {c['pangolin_supervised_tokens']:,} |
| MiniCPM5-2B | {c['minicpm_tokens']:,} | {c['minicpm_supervised_tokens']:,} |

`records` stores full training texts, messages, tools, memory, source provenance and target spans. `tokenized` and `minicpm5_tokenized` store their respective `input_ids` and unshifted assistant-only `labels`; masked positions are `-100`. Causal trainers should shift once. These are aligned views, not extra independent episodes.

```python
from datasets import load_dataset
rows = load_dataset("{DATASET}", "records", split="train", streaming=True)
row = next(iter(rows))
```

The [library and harness](https://github.com/OpenFormosa/barbet-skill-library/tree/{code}) provide source-attributed memory, skill retrieval, typed registered tool callbacks and bounded refinement using real verifier feedback. `prepare_dataset_request` creates answer-free Chat or exact-renderer Completion requests, removing the selected answer and later turns while preserving earlier actions/observations. The JSON tool interface lives in native message content; do not add another tool template. Stored training IDs do not transfer to another tokenizer.

Every retained trajectory passes both Pangolin and official MiniCPM5-2B 128K budgets with 4,096 output tokens reserved, plus measured served-Qwen Chat and Completion budgets. There is no unrelated padding or truncation. The compaction step compares every retained records row and both complete token arrays with the original verified values. Source families and normalized first-user queries do not cross splits, including across waves.

The first pilot reviewed 40 stratified examples (34 ACCEPT, 5 REVISE, 1 REJECT); the second reviewed 17 (13 ACCEPT, 2 REVISE, 2 REJECT). All first non-accept records and same-query variants remain held. These diagnostic samples are not representative accuracy estimates. A contradictory reviewer complaint is retained as a hold, not relabeled ACCEPT. Unreviewed wave-1 search rows remain held. Imported answers and recorded tool observations retain honest upstream verification/origin labels; they are not all independently semantically checked or locally rerun. The declared-parameter guard is additional curation, not a claim that permissive JSON Schema is invalid.

There are {c['assistant_turns']:,} supervised assistant turns and {c['tool_observations']:,} tool observations. {c['actual_retrieval']:,} examples contain actual local deterministic retrieval. `recorded_error_followup` is only an error-word heuristic over tool text, not proof of a real failure, follow-up action or successful repair.

No model weights were trained and no improvement over MiniCPM5 or Qwen3.5-9B is established. Native token-format checks do not certify a Barbet Attention/Mamba vLLM backend. Most imported tasks are English; Traditional Chinese skills do not imply a predominantly Traditional Chinese corpus. Benchmark exclusion covers pinned local exact/13-word matches and specified source exclusions, not universal decontamination.

See [source attribution and licenses](NOTICE.md), [counts](reports/cumulative-0002/summary.json), [audit scope](reports/cumulative-0002/semantic-audit-scopes.json), and the [Traditional Chinese report](reports/cumulative-0002/report.html). `PUBLICATION.json` is added after full anonymous readback of this release.
'''
    (DEST/'README.md').write_text('\n'.join(header)+md)
    rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>' for k,v in c.items() if k.startswith(('domain:','length_bucket:','verification_tier:')))
    body=f'''<h1>Barbet 記憶、技能與工具資料：第二批累積報告</h1><p>目前 {c['accepted']:,} 條合格新軌跡、{u['query_hash']:,} 個不同首題、{u['family_id']:,} 個保守家族；一千萬筆目標完成 {c['accepted']/100000:.2f}%，尚未結案。</p>
<p>第一批提供閱讀文件、真實檢索、數學與工具軌跡；第二批加入有原始上下文的 OpenOrca 任務與經過來源品質篩選、工具順序驗證的 Toucan 軌跡。不同教師對同一來源題目的回答、同一對話的重複版本均去重，兩套 tokenizer 不重複計數。現役舊版 1M 填充資料為零。</p>
<p>第二批抽查揭露 JSON Schema 預設允許額外欄位，不代表實際函式提供那些參數。一條軌跡傳入未宣告的透明背景欄位，並把實際 PNG 回傳說成 SVG。我們保留首次失敗，新增參數宣告檢查，回查並排除第一批 {c['excluded:new_declared_argument_guard']:,} 筆不明確資料。來源 schema、答案與 labels 沒有被改寫來湊過關。</p>
<p>第二批 17 筆診斷為 13 ACCEPT、2 REVISE、2 REJECT。其中一則密碼位數批評本身矛盾；仍保留原 hold，沒有改成 ACCEPT。這也說明 reviewer 不是完美正確率測量器，不能把抽查通過率當成全資料品質。</p>
<table><tr><th>分類</th><th>數量</th></tr>{rows}</table>
<p>Pangolin 有效監督 {c['pangolin_supervised_tokens']:,} tokens；MiniCPM5 有效監督 {c['minicpm_supervised_tokens']:,} tokens。每筆都讀回完整資料與陣列，量測兩個訓練格式及 Qwen 的 Chat/Completion 預算。合併較大 Parquet 檔案只改善載入與傳輸效率，保留的每個值都必須完全一致。</p>
<p>v26 的既有診斷指出：修正格式後數學仍為 0/127、工具完整正確為 4/127。因此資料同時補入基本計算、工具選擇、型別及字面值、外部觀察與證據對應；新 harness 支援真實 callback 與最多兩次 verifier 回饋修正。這些是資料與工程進展，沒有訓練或新模型增益的結論。</p>
<p>後續來源仍需檢查真正題目數、來源重疊、前後文是否包含必要資訊，以及是否把標準答案或未發生的觀察洩漏到輸入。一千萬筆以通過規則的新軌跡計數，持續另報獨立首題與有效監督 token。</p>'''
    (reports/'report.html').write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Barbet v2 累積報告</title><style>body{font:17px/1.8 system-ui;max-width:1000px;margin:40px auto;padding:0 24px}td,th{text-align:left;padding:8px;border-bottom:1px solid #ccd}table{border-collapse:collapse;width:100%}</style>'+body+'</html>')
    manifest=[{'path':str(p.relative_to(DEST)),'bytes':p.stat().st_size,'sha256':file_sha(p)} for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='PUBLICATION.json']
    save(STATE/'publication-plan.json',{'parent_commit':parent,'library_commit':code,'manifest':manifest,'accepted':c['accepted']})
    print(json.dumps({'files':len(manifest),'bytes':sum(f['bytes'] for f in manifest),'accepted':c['accepted']}),flush=True)


def publish():
    if (STATE/'SUPERSEDED.json').exists():raise ValueError('superseded by user v26 minimum-PPL selection')
    plan=json.loads((STATE/'publication-plan.json').read_text());api=HfApi()
    if (STATE/'publication-data-commit.json').exists():raise ValueError('already committed')
    if api.repo_info(DATASET,repo_type='dataset').sha!=plan['parent_commit']:raise ValueError('remote changed')
    for f in plan['manifest']:
        if file_sha(DEST/f['path'])!=f['sha256']:raise ValueError('planned bytes changed')
    paths=api.list_repo_files(DATASET,repo_type='dataset',revision=plan['parent_commit'])
    prefixes=sorted({p.split('/')[0] for p in paths if '/' in p})
    operations=[CommitOperationDelete(p,is_folder=True) for p in prefixes]
    operations += [CommitOperationDelete(p,is_folder=False) for p in paths if '/' not in p and p not in {'.gitattributes','README.md'}]
    operations += [CommitOperationAdd(f['path'],DEST/f['path']) for f in plan['manifest']]
    result=api.create_commit(DATASET,repo_type='dataset',operations=operations,parent_commit=plan['parent_commit'],num_threads=4,
        commit_message=plan.get('commit_message','Publish cumulative memory SFT waves with stricter function arguments and compact exact arrays'))
    save(STATE/'publication-data-commit.json',{'commit':result.oid,'parent':plan['parent_commit'],'plan_sha256':file_sha(STATE/'publication-plan.json')})
    print(json.dumps({'commit':result.oid,'accepted':plan['accepted']}),flush=True)


def verify():
    plan=json.loads((STATE/'publication-plan.json').read_text());data=json.loads((STATE/'publication-data-commit.json').read_text());revision=data['commit']
    if file_sha(STATE/'publication-plan.json')!=data['plan_sha256']:raise ValueError('plan changed')
    folder=STATE/'anonymous-readback/files';folder.mkdir(parents=True,exist_ok=True);checks=[]
    with ThreadPoolExecutor(max_workers=16) as pool:
        tasks=[pool.submit(verify_file,f,revision,folder) for f in plan['manifest']]
        for n,future in enumerate(as_completed(tasks),1):
            checks.append(future.result())
            if n%50==0:print(json.dumps({'verified':n,'total':len(tasks)}),flush=True)
    if set(HfApi(token=False).list_repo_files(DATASET,repo_type='dataset',revision=revision))!=({f['path'] for f in plan['manifest']}|{'.gitattributes'}):raise ValueError('tree mismatch')
    save(STATE/'anonymous-readback/summary.json',{'commit':revision,'checks':sorted(checks,key=lambda x:x['path']),'all_files_verified':True})
    report={'dataset':DATASET,'data_commit':revision,'parent':data['parent'],'active_new_episodes':plan['accepted'],
        'active_legacy_rows':0,'accepted_episode_target':10000000,'goal_complete':False,'anonymous_full_file_sha256_readback':True,
        'verified_files':len(checks),'verified_bytes':sum(f['bytes'] for f in checks),
        'new_declared_argument_holds_removed':json.loads((STATE/'summary.json').read_text())['counts'].get('excluded:new_declared_argument_guard',0),
        'library_commit':plan['library_commit'],'model_efficacy_measured':False,'readback_sha256':file_sha(STATE/'anonymous-readback/summary.json'),
        'answer_selection':plan.get('answer_selection','source-wave selection; no v26 ranking claimed')}
    save(DEST/'PUBLICATION.json',report);api=HfApi()
    metadata_receipt=STATE/'publication-metadata-commit.json'
    if metadata_receipt.exists():
        receipt=json.loads(metadata_receipt.read_text())
        if receipt['data_commit']!=revision or receipt['publication_sha256']!=file_sha(DEST/'PUBLICATION.json'):raise ValueError('metadata resume binding changed')
        final_head=receipt['final_head']
    else:
        if api.repo_info(DATASET,repo_type='dataset').sha!=revision:raise ValueError('head changed before proof publication')
        result=api.create_commit(DATASET,repo_type='dataset',operations=[CommitOperationAdd('PUBLICATION.json',DEST/'PUBLICATION.json')],
            commit_message='Record anonymous verification of cumulative new-only SFT release',parent_commit=revision)
        final_head=result.oid
        save(metadata_receipt,{'data_commit':revision,'final_head':final_head,'publication_sha256':file_sha(DEST/'PUBLICATION.json')})
    final_folder=STATE/'anonymous-readback/metadata';final_folder.mkdir(exist_ok=True)
    verify_file({'path':'PUBLICATION.json','bytes':(DEST/'PUBLICATION.json').stat().st_size,'sha256':file_sha(DEST/'PUBLICATION.json')},final_head,final_folder)
    save(STATE/'publication-final-commit.json',{**report,'final_head':final_head})
    save(FIRST/'GOAL_PROGRESS.json',{'target':10000000,'published_eligible_new_episodes':plan['accepted'],'phase':plan.get('goal_phase','two_waves_published_continuing_expansion'),'goal_complete':False,'final_head':final_head})
    print(json.dumps({'head':final_head,'accepted':plan['accepted'],'goal_complete':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','publish','verify']);args=p.parse_args()
    with (FIRST/'publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.phase]()
