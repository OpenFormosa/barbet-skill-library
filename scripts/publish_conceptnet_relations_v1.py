"""Append checked lexical-memory tasks; preserve all prior rows and holds."""
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
from scripts.publish_wikidata_relations_v1 import STATE as PARENT,DEST as PARENT_DEST
from scripts.package_conceptnet_relations_v1 import STATE as SELECTED,PUBLIC,SOURCE
from scripts import normalize_conceptnet_relations_v1 as normalizer
from scripts.index_conceptnet_relations_v1 import STATE as SCORE
from scripts.render_memory_harness_card_v3 import render
from scripts.conceptnet_source_notices_v1 import write_notices

STATE=SELECTED/'cumulative-publication-v1'
DEST=base.WORK/'exports/memory-harness-v2-cumulative-0010'
FIRST=base.WORK/'reports/memory-harness-v2'
WAVE='cumulative-0010'


def prepare():
    STATE.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True)
    if (STATE/'publication-plan.json').exists():raise ValueError('lexical publication plan already frozen')
    resources=json.loads(normalizer.RESOURCES.read_text())['resources'];reserve=5*1024**3
    if normalizer.processed_bytes()+reserve>resources['new_processed_cap_bytes'] or shutil.disk_usage(base.WORK).free<resources['minimum_free_disk_bytes']+reserve:
        raise ValueError('cumulative lexical metadata resource gate')
    proof=json.loads((PARENT/'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']:raise ValueError('previous cumulative release unverified')
    prior_plan=json.loads((PARENT/'publication-plan.json').read_text());prior=json.loads((PARENT/'summary.json').read_text())
    new=json.loads((SELECTED/'public-package-summary.json').read_text())
    for key in ['all_three_configs_exact_readback','all_answer_free_qwen_chat_and_completion_budgets_measured',
        'all_native_budgets_include_4096_output_reserve','all_previous_folds_preserved','no_published_question_overlap',
        'first_nonaccept_query_holds_preserved','all_budget_proof_identities_equal','all_records_exact_finite_query_SQL_verified']:
        if not new[key]:raise ValueError('new lexical release gate missing: '+key)
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
        raise ValueError('incoming lexical question overlaps previous release')
    db.execute('CREATE TABLE records AS SELECT * FROM previous.records')
    db.execute('INSERT INTO records SELECT * FROM incoming.records')
    db.execute('CREATE UNIQUE INDEX record_identity ON records(record_id)');db.commit()
    for key in ['query_hash','family_id']:
        if db.execute(f'SELECT 1 FROM records GROUP BY {key} HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone():
            raise ValueError('cumulative split leakage')
    old_holds=json.loads((PARENT_DEST/'reports/cumulative-0006/query-wide-holds.json').read_text())['query_hashes']
    new_holds=json.loads((SOURCE/'first-review-query-holds.json').read_text())['query_hashes']
    prior_graph_holds=json.loads((PARENT_DEST/'reports/cumulative-0009/graph-first-query-holds.json').read_text())['query_hashes']
    holds=old_holds+prior_graph_holds+new_holds
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
    copies=[(SELECTED/'ppl-score-binding.json','lexical-singleton-binding.json'),
        (SELECTED/'selection-policy.json','lexical-selection-policy.json'),
        (SELECTED/'release-scope.json','lexical-release-scope.json'),
        (SELECTED/'package-policy-v1.json','lexical-package-policy.json'),
        (SELECTED/'dedup-summary.json','lexical-selection-summary.json'),
        (SELECTED/'compile-summary.json','lexical-compile-summary.json'),
        (SELECTED/'framed-compile-policy.json','lexical-compile-policy.json'),
        (normalizer.STATE/'normalization-policy.json','lexical-source-policy.json'),
        (normalizer.STATE/'normalize-summary.json','lexical-source-summary.json'),
        (SCORE/'candidate-policy-binding.json','lexical-index-policy.json'),
        (SCORE/'candidate-index-summary.json','lexical-index-summary.json'),
        (SCORE/'episode-identity-v1/summary.json','lexical-episode-identity-summary.json'),
        (SOURCE/'pilot-summary.json','lexical-first-source-pilot.json'),
        (SOURCE/'semantic-audit-v1/summary.json','lexical-first-source-diagnosis.json'),
        (SOURCE/'first-review-query-holds.json','lexical-first-query-holds.json'),
        (SOURCE/'independent-sql-proofs.json','lexical-pilot-independent-sql.json'),
        (source_root/'export-probe-v1/first-review-independent-analysis.json','lexical-first-review-independent-analysis.json'),
        (source_root/'source-plan.json','lexical-primary-source-plan.json'),
        (source_root/'source-lock.json','lexical-primary-source-lock.json'),
        (source_root/'framed-compile-probe-v1/summary.json','lexical-direct-frame-compile-equivalence.json'),
        (source_root/'export-probe-v1/summary.json','lexical-portable-export-equivalence.json'),
        (normalizer.adapter.INVENTORY/'summary.json','lexical-labeled-source-inventory.json'),
        (normalizer.adapter.INVENTORY/'policy.json','lexical-source-scope-policy.json'),
        (normalizer.adapter.ORIGINAL/'summary.json','lexical-full-English-source-inventory.json'),
        (FIRST/'relational-memory-tests-v1.log','relational-memory-tests-v1.log'),
        (FIRST/'relational-dataset-tests-v1.log','relational-dataset-tests-v1.log'),
        (normalizer.RESOURCES,'prospective-resources-v3.json'),
        (PARENT_DEST/'PUBLICATION.json','prior-publication.json')]
    for source,name in copies:shutil.copyfile(source,folder/name)
    notice='''

## ConceptNet 5.7 lexical-memory addition

Original source and attribution: https://github.com/commonsense/conceptnet5/wiki/Downloads and https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet . Original dump: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz . This historical snapshot is not current-world factual certification. Retained source datasets are /d/wiktionary/en, /d/wiktionary/fr, /d/wiktionary/de and /d/wordnet/3.1, restricted to English concept URIs at both ends. Other source families, including OMCS and Verbosity, were excluded prospectively.

ConceptNet attribution, every assertion URI, source dataset, contributing source records, metadata and explicit per-assertion CC-BY-SA-4.0 or CC-BY-4.0 license remain in source_refs_json. Adapted rows containing any CC-BY-SA-4.0 assertion are marked CC-BY-SA-4.0; rows containing only CC-BY-4.0 assertions retain that license. ShareAlike obligations remain applicable. These records are not relicensed under the repository's Apache-2.0 license. Concept URIs preserve sense/POS suffixes and relation polarity. Source licenses: https://creativecommons.org/licenses/by-sa/4.0/ and https://creativecommons.org/licenses/by/4.0/ .

Adaptations: one deterministic bounded relation query per base concept, one fixed front/back/tool placement, exact set/count targets, independent SQLite recomputation, actual registered read-only tool observations, skill instructions and assistant-only native token labels. Multiple qualified URI variants of one base concept do not create additional independent questions. No downloaded programs were executed; no teacher-generated answer prose or invented PPL scores were added. Source assertions are evidence within a finite provided snapshot, not independently verified world truth.
'''
    notice+=write_notices(DEST,folder)
    (DEST/'NOTICE.md').write_text((PARENT_DEST/'NOTICE.md').read_text()+notice)
    note=f'''This addition contains **{new['counts']['accepted']:,}** independently checked lexical-memory queries. The complete original dump contains 34,074,917 source rows. The prospective Wiktionary/WordNet scope retains 2,847,001 assertions. Its 1,303,855 subject URIs collapse into 871,550 base concepts, avoiding 432,305 additional URI-variant views. Complete construction excluded 283 anchors above adjacency budgets and all seven first non-ACCEPT anchors, leaving 871,260 candidates. All 488,568 recorded tool observations were restored from portable facts. The complete candidate index found no alternative-answer groups; retained singletons are explicitly unscored.

The first source diagnosis was 18 ACCEPT and 7 REVISE across 25 distinct base-concept samples. Every sample target matched independent SQL, and every actual reviewer request matched the complete saved messages. Some negative critiques contradicted valid JSON or evidence IDs already present, or stated that the answer was correct while selecting REVISE. All seven questions remain held without replay or replacement; this is not a corpus error rate. A separate 1,024-row exact export probe and two real reading/tool framed-compilation cases checked full text, arrays, labels and portable state. Per-assertion source metadata and CC-BY/CC-BY-SA licenses remain explicit. No source assertion is certified as current-world truth.

The prior 4,547,341 Wikidata structured-memory rows and all earlier releases are retained byte-for-byte. Together the two procedural relation cohorts use a small set of query schemas over distinct source anchors; their row count is not a claim of equally many distinct reasoning algorithms or benchmark improvements.'''
    card=render(counts,unique,code,WAVE,new['counts']['accepted'],note)
    (DEST/'README.md').write_text(card);DatasetCard(card).validate()
    body=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>一千萬筆記憶與技能訓練資料</title><style>body{{font:17px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:24px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><h1>新資料擴充：完整計數與驗收</h1><p>本批新增 {new['counts']['accepted']:,} 筆；累計 {counts['accepted']:,} 筆新 episode、{unique['query_hash']:,} 個不同標準化首問、{unique['family_id']:,} 個任務家族，現役舊資料為 0。資料量目標以通過檢查的 episode 計算，不使用答案版本、記憶位置或詞義 URI 變體湊數。遠端完整回讀狀態另見 PUBLICATION.json。</p><p>同一完整輸入的多個合格答案依固定 native v26 最低條件 PPL 選取；本詞典批次每題只有一個精確目標，全部明確標為未量測 PPL。圖譜／詞典答案逐筆以獨立 SQL 驗證，工具觀察可從保存的唯讀 facts 還原。第一輪語意診斷 18 通過、7 需修正；7 題持續隔離，不以再次抽查覆寫原判決。</p><p>結構化查詢資料使用少數程序格式搭配不同實體或詞彙，因此原始筆數不能當作能力增益。來源、題型及有效 supervised tokens 應分別控制訓練比例。原 coding 樣本沒有執行來源程式，沒有公開回饋的列不假稱 iterative refinement。這次工作沒有訓練模型，沒有量測新資料使小模型追上 9B。</p><p>逐條來源與 CC-BY/CC-BY-SA 授權均保留；前後文與觀察的 loss 被遮罩。完整文字、兩種 tokenizer 的陣列與目標標籤均回讀核對，兩種答案隱藏請求預算逐題量測，新舊家族及問題切分保持一致。</p><pre>{html.escape(json.dumps({k:v for k,v in summary.items() if k!='files'},ensure_ascii=False,indent=2))}</pre></html>'''
    (folder/'report.html').write_text(body)
    manifest=[{'path':str(p.relative_to(DEST)),'bytes':p.stat().st_size,'sha256':base.file_sha(p)}
        for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='PUBLICATION.json']
    old={f['path']:f['sha256'] for f in prior_plan['manifest']}
    base.save(STATE/'publication-plan.json',{'parent_commit':proof['final_head'],'parent_plan_sha256':base.file_sha(PARENT/'publication-plan.json'),
        'library_commit':code,'manifest':manifest,'accepted':counts['accepted'],
        'incremental_add_paths':[f['path'] for f in manifest if old.get(f['path'])!=f['sha256']],
        'answer_selection':'minimum eligible conditional answer PPL under frozen native Barbet v26; singleton answers explicitly unscored',
        'goal_phase':'ten_million_new_episodes_published_with_exact_memory_queries'})
    print(json.dumps({'accepted':counts['accepted'],'new_rows':new['counts']['accepted'],'files':len(manifest)}),flush=True)


def publish():
    if (STATE/'publication-data-commit.json').exists():raise ValueError('lexical wave already committed')
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
        if args.phase=='verify':
            from scripts.verify_memory_target_release_v1 import main as verify_target
            verify_target(STATE,DEST)
        else:globals()[args.phase]()
