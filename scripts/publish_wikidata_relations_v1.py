"""Append checked structured-memory tasks; preserve all prior rows and holds."""
import argparse
from collections import Counter
import fcntl
import html
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

from huggingface_hub import DatasetCard,HfApi,CommitOperationAdd
from scripts import build_memory_sft_v2 as base
from scripts import publish_memory_cumulative_v2 as transport
from scripts.publish_natural_instructions_v26_v1 import link_checked
from scripts.publish_ifm_code_v26_v1 import STATE as PARENT,DEST as PARENT_DEST
from scripts.package_wikidata_relations_v1 import STATE as SELECTED,PUBLIC,SOURCE
from scripts import normalize_wikidata_relations_v1 as normalizer
from scripts.index_wikidata_relations_v1 import STATE as SCORE
from scripts.render_memory_harness_card_v3 import render

STATE=SELECTED/'cumulative-publication-v1'
DEST=base.WORK/'exports/memory-harness-v2-cumulative-0009'
FIRST=base.WORK/'reports/memory-harness-v2'
WAVE='cumulative-0009'


def prepare():
    STATE.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True)
    if (STATE/'publication-plan.json').exists():raise ValueError('graph publication plan already frozen')
    resources=json.loads(normalizer.RESOURCES.read_text())['resources'];reserve=5*1024**3
    if normalizer.processed_bytes()+reserve>resources['new_processed_cap_bytes'] or shutil.disk_usage(base.WORK).free<resources['minimum_free_disk_bytes']+reserve:
        raise ValueError('cumulative graph metadata resource gate')
    proof=json.loads((PARENT/'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']:raise ValueError('previous cumulative release unverified')
    prior_plan=json.loads((PARENT/'publication-plan.json').read_text());prior=json.loads((PARENT/'summary.json').read_text())
    new=json.loads((SELECTED/'public-package-summary.json').read_text())
    for key in ['all_three_configs_exact_readback','all_answer_free_qwen_chat_and_completion_budgets_measured',
        'all_native_budgets_include_4096_output_reserve','all_previous_folds_preserved','no_published_question_overlap',
        'first_nonaccept_query_holds_preserved','all_budget_proof_identities_equal','all_records_exact_finite_query_SQL_verified']:
        if not new[key]:raise ValueError('new graph release gate missing: '+key)
    for item in prior_plan['manifest']:
        if item['path'] not in ['README.md','NOTICE.md','PUBLICATION.json']:
            link_checked(PARENT_DEST/item['path'],DEST/item['path'],item['sha256'])
    for item in new['files']:link_checked(PUBLIC/item['path'],DEST/item['path'],item['sha256'])
    index=STATE/'public-index.sqlite'
    if index.exists():raise ValueError('partial cumulative index exists; preserve and diagnose')
    db=sqlite3.connect(index);db.execute('PRAGMA journal_mode=OFF')
    db.execute('ATTACH DATABASE ? AS previous',(str(PARENT/'public-index.sqlite'),))
    db.execute('ATTACH DATABASE ? AS incoming',(str(SELECTED/'public-index.sqlite'),))
    if db.execute('SELECT 1 FROM incoming.records JOIN previous.records USING(query_hash) LIMIT 1').fetchone():
        raise ValueError('incoming graph question overlaps previous release')
    db.execute('CREATE TABLE records AS SELECT * FROM previous.records')
    db.execute('INSERT INTO records SELECT * FROM incoming.records')
    db.execute('CREATE UNIQUE INDEX record_identity ON records(record_id)');db.commit()
    for key in ['query_hash','family_id']:
        if db.execute(f'SELECT 1 FROM records GROUP BY {key} HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone():
            raise ValueError('cumulative split leakage')
    old_holds=json.loads((PARENT_DEST/'reports/cumulative-0006/query-wide-holds.json').read_text())['query_hashes']
    new_holds=json.loads((SOURCE/'first-review-query-holds.json').read_text())['query_hashes']
    holds=old_holds+new_holds
    if db.execute('SELECT 1 FROM records WHERE query_hash IN ('+','.join('?' for _ in holds)+') LIMIT 1',holds).fetchone():
        raise ValueError('first source or old tool-contract hold reintroduced')
    unique={key:db.execute(f'SELECT COUNT(DISTINCT {key}) FROM records').fetchone()[0] for key in ['record_id','query_hash','family_id']}
    db.close();counts=Counter(prior['counts']);counts.update(new['counts'])
    if counts['accepted']!=unique['record_id']:raise ValueError('cumulative row count differs')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip()
    summary={'wave':WAVE,'counts':dict(counts),'unique_counts':unique,'files':prior['files']+new['files'],
        'new_rows':new['counts']['accepted'],'prior_release':proof['final_head'],'library_commit':code,
        'new_wave':{k:v for k,v in new.items() if k!='files'},'prior_rows_preserved_byte_for_byte':True,
        'all_cumulative_folds_preserved':True,'all_previous_source_contract_query_holds_preserved':True,
        'accepted_episode_target':10000000,'goal_complete':False,'model_weights_trained':False,'model_efficacy_measured':False,
        'status':'PAYLOAD_VALIDATED; REMOTE_READBACK_REPORTED_SEPARATELY_IN_PUBLICATION_JSON'}
    base.save(STATE/'summary.json',summary);folder=DEST/'reports'/WAVE;folder.mkdir(parents=True,exist_ok=True)
    base.save(folder/'summary.json',{k:v for k,v in summary.items() if k!='files'});base.save(folder/'data-files.json',summary['files'])
    source_root=normalizer.STATE.parent
    copies=[(SELECTED/'ppl-score-binding.json','graph-singleton-binding.json'),
        (SELECTED/'selection-policy.json','graph-selection-policy.json'),
        (SELECTED/'release-scope.json','graph-release-scope.json'),
        (SELECTED/'package-policy-v1.json','graph-package-policy.json'),
        (SELECTED/'dedup-summary.json','graph-selection-summary.json'),
        (SELECTED/'compile-summary.json','graph-compile-summary.json'),
        (SELECTED/'framed-compile-policy.json','graph-compile-policy.json'),
        (normalizer.STATE/'normalization-policy.json','graph-source-policy.json'),
        (normalizer.STATE/'normalize-summary.json','graph-source-summary.json'),
        (SCORE/'candidate-policy-binding.json','graph-index-policy.json'),
        (SCORE/'candidate-index-summary.json','graph-index-summary.json'),
        (SCORE/'episode-identity-v1/summary.json','graph-episode-identity-summary.json'),
        (SOURCE/'pilot-summary.json','graph-first-source-pilot.json'),
        (SOURCE/'semantic-audit-v1/summary.json','graph-first-source-diagnosis.json'),
        (SOURCE/'first-review-query-holds.json','graph-first-query-holds.json'),
        (SOURCE/'independent-sql-proofs.json','graph-pilot-independent-sql.json'),
        (SOURCE/'first-review-error-analysis.json','graph-first-review-independent-analysis.json'),
        (source_root/'source-plan.json','graph-primary-source-plan.json'),
        (source_root/'source-lock.json','graph-primary-source-lock.json'),
        (source_root/'framed-compile-probe-v1/summary.json','graph-direct-frame-compile-equivalence.json'),
        (source_root/'export-probe-v1/summary.json','graph-portable-export-equivalence.json'),
        (normalizer.adapter.INVENTORY/'summary.json','graph-labeled-source-inventory.json'),
        (normalizer.adapter.INVENTORY/'alias-coverage-exclusions.json','graph-source-label-coverage.json'),
        (FIRST/'relational-memory-tests-v1.log','relational-memory-tests-v1.log'),
        (FIRST/'relational-dataset-tests-v1.log','relational-dataset-tests-v1.log'),
        (normalizer.RESOURCES,'prospective-resources-v3.json'),
        (PARENT_DEST/'PUBLICATION.json','prior-publication.json')]
    for source,name in copies:shutil.copyfile(source,folder/name)
    notice='''

## Wikidata5M structured relation addition

Source: https://deepgraphlearning.github.io/project/wikidata5m and https://github.com/THU-KEG/KEPLER . Only the original author-provided inductive TRAIN structured triples and entity/relation aliases were used. Wikidata structured data is CC0: https://www.wikidata.org/wiki/Wikidata:Licensing . Wikipedia article text was not downloaded or relicensed. This historical snapshot is not a claim about current-world facts. Original validation/test graph entities were excluded. Missing source aliases caused incident facts to be excluded instead of inventing labels.

Adaptations: one deterministic bounded relation-query task per eligible training subject; one fixed front/back/tool placement; exact project/count/set targets; independent SQLite verification; causal registered read-only lookup observations; skill instructions; exact native text, arrays and labels. Full per-record source line numbers, source member hash, original IDs/labels and attribution are retained. General query schemas are reported separately from independently anchored source questions. All eligible complete input groups were proven singleton; no PPL score or learned model result is claimed for these exact targets. Family splitting can share other graph facts across SFT splits and is not entity-disjoint evaluation.
'''
    (DEST/'NOTICE.md').write_text((PARENT_DEST/'NOTICE.md').read_text()+notice)
    note=f'''This addition contains **{new['counts']['accepted']:,}** exact finite-record queries from the original Wikidata5M training graph, under CC0 for structured data. All 4,554,029 eligible source anchors were inspected; 6,673 exceeded complete-adjacency budgets and 15 first non-ACCEPT source anchors remain held. The normalized cohort has 4,547,341 candidates, each independently checked using SQLite. Global identity checks found zero alternative-answer groups. These are explicitly unscored singletons, not v26 PPL winners. Actual read-only callback observations are preserved and can be restored from portable snapshot metadata.

The first source diagnosis had 21 ACCEPT and 15 REVISE across 36 distinct sampled anchors. All 36 exact SQL targets matched before review. Some negative reviewer complaints contradicted the saved full request, such as claiming a present record was missing or valid JSON was truncated. All 15 first holds remain excluded without review replay or replacement. These counts are not an error rate for the full graph. Every row has an exact finite-record target verification; this is separate from world-truth or general natural-language semantic certification. No Wikipedia article text was used. The source's inductive validation/test entities were excluded, but this dataset's new anchor-based SFT splits may share context facts.'''
    card=render(counts,unique,code,WAVE,new['counts']['accepted'],note)
    (DEST/'README.md').write_text(card);DatasetCard(card).validate()
    body=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>圖譜記憶與工具資料發布</title><style>body{{font:17px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:24px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><h1>逐題驗證的圖譜記憶資料</h1><p>新增 {new['counts']['accepted']:,} 筆；累計 {counts['accepted']:,} 筆新資料。所有圖譜題目均為單一合格答案，未量測 PPL；多答案題仍使用既定 v26 最低條件 PPL 規則。圖譜答案逐筆由獨立 SQL 重算，工具觀察由實際唯讀查詢產生，前後記憶不重複計數。</p><p>首次來源抽查 21 通過、15 需修正；15 題全部持續隔離。部分批評與完整請求矛盾，不能把診斷票數當成全資料錯誤率。精確查詢驗證不等於現實世界事實或模型能力證明。未訓練模型，未證明小模型達到 9B 水準。</p><p>全部三種輸出逐列回讀，核對完整 token 與 label 陣列，以及兩種答案隱藏推論請求的預算。來源使用原始訓練圖譜；SFT 依 anchor 家族切分，其他上下文實體可能共用，不宣稱 entity-disjoint。發布遠端回讀結果另見 PUBLICATION.json。</p><pre>{html.escape(json.dumps({k:v for k,v in summary.items() if k!='files'},ensure_ascii=False,indent=2))}</pre></html>'''
    (folder/'report.html').write_text(body)
    manifest=[{'path':str(p.relative_to(DEST)),'bytes':p.stat().st_size,'sha256':base.file_sha(p)}
        for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='PUBLICATION.json']
    old={f['path']:f['sha256'] for f in prior_plan['manifest']}
    base.save(STATE/'publication-plan.json',{'parent_commit':proof['final_head'],'parent_plan_sha256':base.file_sha(PARENT/'publication-plan.json'),
        'library_commit':code,'manifest':manifest,'accepted':counts['accepted'],
        'incremental_add_paths':[f['path'] for f in manifest if old.get(f['path'])!=f['sha256']],
        'answer_selection':'minimum eligible conditional answer PPL under frozen native Barbet v26; singleton answers explicitly unscored',
        'goal_phase':'wikidata_exact_memory_queries_published_continuing_to_10m'})
    print(json.dumps({'accepted':counts['accepted'],'new_rows':new['counts']['accepted'],'files':len(manifest)}),flush=True)


def publish():
    if (STATE/'publication-data-commit.json').exists():raise ValueError('graph wave already committed')
    plan=json.loads((STATE/'publication-plan.json').read_text());api=HfApi()
    if base.file_sha(PARENT/'publication-plan.json')!=plan['parent_plan_sha256']:raise ValueError('parent manifest changed')
    if api.repo_info(transport.DATASET,repo_type='dataset').sha!=plan['parent_commit']:raise ValueError('remote head changed')
    actual=set(HfApi(token=False).list_repo_files(transport.DATASET,repo_type='dataset',revision=plan['parent_commit']))
    previous=json.loads((PARENT/'publication-plan.json').read_text())
    if actual!={f['path'] for f in previous['manifest']}|{'.gitattributes','PUBLICATION.json'}:raise ValueError('parent tree differs')
    old={f['path']:f['sha256'] for f in previous['manifest']};changed={f['path'] for f in plan['manifest'] if old.get(f['path'])!=f['sha256']}
    if changed!=set(plan['incremental_add_paths']):raise ValueError('incremental scope differs')
    operations=transport.replacement_deletions(actual,[f['path'] for f in plan['manifest']])
    for item in plan['manifest']:
        if base.file_sha(DEST/item['path'])!=item['sha256']:raise ValueError('planned payload changed')
        if item['path'] in changed:operations.append(CommitOperationAdd(item['path'],DEST/item['path']))
    commit=api.create_commit(transport.DATASET,repo_type='dataset',operations=operations,parent_commit=plan['parent_commit'],
        commit_message='Append independently verified finite-record memory and tool queries; preserve frozen v26 alternative-answer rule',num_threads=8)
    base.save(STATE/'publication-data-commit.json',{'commit':commit.oid,'parent':plan['parent_commit'],'plan_sha256':base.file_sha(STATE/'publication-plan.json')})
    print(json.dumps({'commit':commit.oid,'accepted':plan['accepted']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','publish','verify']);args=parser.parse_args()
    with (FIRST/'publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        transport.STATE=STATE;transport.DEST=DEST;transport.FIRST=FIRST
        if args.phase=='verify':transport.verify()
        else:globals()[args.phase]()
