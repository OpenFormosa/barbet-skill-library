"""Update public repository pointers only after final anonymous verification."""
import datetime
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi
from scripts import publish_taiwan_civic_v1 as release


def main():
    proof=release.read(release.STATE/'publication-final-commit.json')
    if not proof['publication_receipt_anonymous_verified']:raise ValueError('Final proof not verified')
    head=proof['final_head']
    if HfApi(token=False).repo_info(release.DATASET,repo_type='dataset').sha!=head:raise ValueError('Main advanced; review before changing pointers')
    repo=release.prior.parent.base.REPO;report=repo/'docs/reports/taiwan-civic-v1';report.mkdir(exist_ok=True)
    shutil.copyfile(release.DEST/release.FOLDER/'report.html',report/'index.html')
    release.save(report/'publication.json',proof)
    recipe=release.read(release.DEST/'recipe/barbet-training-v2.json');recipe['dataset_revision']=head
    release.save(repo/'configs/barbet_training_data_v2.json',recipe)
    summary=release.read(release.STATE/'summary.json');c=summary['counts'];u=summary['unique_counts']
    path=repo/'README.md';text=path.read_text();start=text.index('[HF 現行版本]');end=text.index('\n\n**同一完整輸入',start)
    paragraph=f'''[HF 現行版本](https://huggingface.co/datasets/{release.DATASET}/tree/{head}) 以 **Barbet** 為訓練目標：**{c['accepted']:,} 筆新 episode**、**{u['query_hash']:,} 個標準化首問**、**{u['family_id']:,} 個任務家族**，現役舊資料為 0。預設 `tokenized` 使用與 v26 相同的 Barbet tokenizer 和 assistant-only labels；另提供 `records`，不需要 MiniCPM tokenizer。固定版本與 collator 設定見 [續訓設定 v2](configs/barbet_training_data_v2.json)，先前完整格式與 PPL 查核見 [v26 品質報告](docs/reports/v26-continuation-readiness-v1/index.html)。

本輪先移除品質待審整題的 7,552 筆，再隔離政治主題待審的 176 筆、加入 933 筆公民資料，沒有以次低 PPL 答案替補。新資料包含 22 部台灣法規的 920 個條文檢索題、12 個明示價值案例及 1 個有日期與機關歸屬的官方引述；採用台灣主權、人民自決及普世人權，不預設支持政黨。法規摘錄不等於目前個案適用，政治詞彙分流不等於全語料的語義或價值認證。來源、授權、切分、前後 memory 及限制見 [公民資料與品質完整報告](docs/reports/taiwan-civic-v1/index.html)。

新批次全部 933 筆完成原始來源對照、完整 tokens/labels 重建及 ContextAgent 答案前綴一致性檢查；library 共 65 個技能，完整檢查 274 個測試通過。這批僅占全資料約 **0.0031% 的監督 tokens**；HF 的 `curriculum/taiwan-civic-v1.parquet` 提供分層抽樣索引，但抽樣權重與模型效果尚未校準。發布採固定 staging 完整匿名 SHA-256 回讀、保留陣列逐項相等與既有 Git 物件驗證鏈；不是重新下載所有歷史資料。未更新模型權重，也未宣稱政治價值或能力已提升。'''
    path.write_text(text[:start]+paragraph+text[end:])
    checkpoint=release.prior.parent.base.WORK/'reports/memory-harness-v2/WORKING_CHECKPOINT.json'
    state=release.read(checkpoint);state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    state['phase']='quality and Taiwan civic revision published and anonymously verified; no training performed'
    state['public_release']=proof;state['current_quality_audit']={
        'first_quality_report':str(release.prior.STATE),'political_report':str(release.STATE),
        'native_no_grad_smoke_passed':True,'short_forward_artifact':str(release.prior.audit.STATE/'native-collated-forward-v4.json'),
        'old_transport_stop_preserved':str(release.STATE/'transport-v1-controlled-boundary.json'),
        'new_records_source_and_live_prefix_checks':933,'library_unit_tests_passed':274,
        'not_gradient_long_sequence_or_model_improvement_validation':True}
    state['political_data_requirement']={
        'confirmed':True,'stance':'Taiwan sovereignty and peoples free democratic self-determination; universal rights; no default party',
        'status':'Published 933 source-grounded/explicitly normative episodes and the policy/skill; 176 prior episodes conservatively held across 140 queries',
        'normative_case_count':12,'raw_supervised_token_share':recipe['civic_raw_supervised_token_share'],
        'exhaustive_semantic_or_model_alignment_guarantee':False}
    state['known_identity_registry']=str(release.STATE/'known-identity-composition.json')
    release.save(checkpoint,state)
    print(json.dumps({'verified_hf_head':head,'repo_docs_updated':True,'model_weights_trained':False}),flush=True)


if __name__=='__main__':main()
