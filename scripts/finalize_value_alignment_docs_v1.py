"""Pin documentation and the v26 recipe only after final publication readback."""
from datetime import datetime,timezone
import json
import shutil
from huggingface_hub import HfApi
from scripts import publish_value_alignment_v1 as release


def main():
    proof=release.read(release.STATE/'publication-final-commit.json');head=proof['final_head']
    if not proof['publication_receipt_anonymous_verified']:raise ValueError('Final proof is not anonymously verified')
    if HfApi(token=False).repo_info(release.DATASET,repo_type='dataset').sha!=head:raise ValueError('Remote main has advanced')
    report=release.REPO/'docs/reports/value-alignment-v1';report.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(release.DEST/release.FOLDER/'report.html',report/'index.html');release.save(report/'publication.json',proof)
    release.save(report/'admission-summary.json',release.read(release.DEST/release.FOLDER/'admission-summary.json'))
    recipe=release.read(release.DEST/'recipe/barbet-training-v2.json');recipe['dataset_revision']=head
    release.save(release.REPO/'configs/barbet_training_data_v2.json',recipe)
    s=release.read(release.STATE/'summary.json');c=s['counts'];u=s['unique_counts'];a=release.read(release.STATE/'admission.json')
    path=release.REPO/'README.md';text=path.read_text();start=text.index('[HF 現行版本]');end=text.index('\n\n**同一完整輸入',start)
    paragraph=f'''[HF 現行版本](https://huggingface.co/datasets/{release.DATASET}/tree/{head}) 以 **Barbet** 為訓練目標：**{c['accepted']:,} 筆新 episode**、**{u['query_hash']:,} 個標準化首問**、**{u['family_id']:,} 個任務家族**，現役舊資料為 0。預設 `tokenized` 提供 v26 相同的 Barbet tokenizer 與 assistant-only labels，另有 `records`，不需要 MiniCPM tokenizer。固定版本與 collator 見 [續訓設定 v2](configs/barbet_training_data_v2.json)。

本輪對 10,321,901 筆現役資料的實際對話、情境指令及工具定義做全量主題分流，找到 {a['candidate_rows']:,} 筆候選。逐筆驗證 241,552 筆有限來源 ID／計數查詢及 920 筆法律摘錄的完整契約，再核對系統指令與完整資料身份。12 個正反審核控制通過；{a['semantic_first_reviews']} 筆固定語境樣本的初次判定為 {', '.join(f"{key} {value}" for key,value in a['semantic_counts'].items())}。敏感待審及未通過資料整題隔離，共新增移除 {s['new_held_rows']:,} 筆、{s['new_held_queries']:,} 個 query，沒有以次低 PPL 答案替補。待審隔離不表示每筆都錯誤，樣本結果不是全語料錯誤率。

[價值政策](configs/value-alignment-policy-v1.json) 明示支持台灣主權、人民自決、民主、自由、人權、法治、平等與多元，不預設政黨，保障和平異議與對台灣政府的批評，也尊重中國人民及所有群體。忠實引文、翻譯、歷史與不同意見不因立場而改寫。品質與價值納入門檻優先於 v26 PPL。完整範圍、首次判定及限制見 [價值對齊資料查核報告](docs/reports/value-alignment-v1/index.html)。

現役公民資料 {s['retained_civic_episodes']} 筆，原始監督 token 占比約 **{100*recipe['civic_raw_supervised_token_share']:.6f}%**；現役抽樣索引與 refinement 索引均已排除本輪隔離 ID。保留資料的完整 tokens/labels、答案、來源、分割及既有分數完全不變。發布使用固定 staging 匿名 SHA-256 回讀與既有物件驗證鏈。這不是所有一千萬筆的逐筆語義價值認證，也不是模型對齊成效；本輪未訓練或更改權重。先前的 [v26 品質報告](docs/reports/v26-continuation-readiness-v1/index.html) 與 [公民資料報告](docs/reports/taiwan-civic-v1/index.html) 保留為歷史證據。'''
    path.write_text(text[:start]+paragraph+text[end:])
    checkpoint=release.prior.prior.parent.base.WORK/'reports/memory-harness-v2/WORKING_CHECKPOINT.json';state=release.read(checkpoint)
    state.update(updated_at=datetime.now(timezone.utc).isoformat(),latest_user_steering='確保資料是價值對齊的',phase='Values admission revision published and anonymously verified; no model training',public_release=proof)
    state['current_value_alignment']={'policy':str(release.REPO/'configs/value-alignment-policy-v1.json'),'report':str(release.STATE),'candidate_rows':a['candidate_rows'],'new_held_rows':s['new_held_rows'],'new_held_queries':s['new_held_queries'],'all_first_outcomes_preserved':True,'semantic_counts':a['semantic_counts'],'exhaustive_semantic_guarantee':False,'model_alignment_efficacy_measured':False}
    state['political_data_requirement'].update(status='Published all-source values admission with exact source-bound contracts and contextual diagnostics; pending sensitive queries quarantined',raw_supervised_token_share=recipe['civic_raw_supervised_token_share'])
    # Identity registries remain immutable; append the cumulative hold source.
    state['current_value_hold_registry']=str(release.STATE/'admission.json')
    release.save(checkpoint,state);print(json.dumps({'head':head,'active':c['accepted'],'repository_pointers_updated':True}),flush=True)


if __name__=='__main__':main()
