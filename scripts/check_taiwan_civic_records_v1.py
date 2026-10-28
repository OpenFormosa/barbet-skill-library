"""Independently re-read the real civic cohort against original source texts."""
from collections import Counter
import json
import re
import xml.etree.ElementTree as ET

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from barbet_inference.dataset_adapter import prepare_dataset_request
from barbet_inference.sft import pangolin_render, tokenize_spans
from scripts.build_taiwan_civic_v1 import STATE, DEST, RAW, base, prior


def main():
    output = STATE / 'independent-record-check.json'
    if output.exists(): raise ValueError('First full cohort check already recorded')
    summary = prior.read(STATE / 'new-cohort-summary.json')
    root = ET.parse(RAW / 'laws.xml').getroot()
    originals = {}
    for law in root:
        for a in law.findall('.//條文'):
            originals[(law.findtext('法規名稱'), a.findtext('條號'))] = a.findtext('條文內容', '').strip()
    tokenizer = Tokenizer.from_file(str(base.WORK / 'data/blobs/tokenizer/tokenizer.json'))
    counts = Counter(); families = {}; records = []
    for item in summary['files']:
        if '/records/' not in item['path']: continue
        path = DEST / item['path']; token_path = DEST / item['path'].replace('/records/', '/tokenized/')
        if prior.sha(path) != item['sha256']: raise ValueError('Record file changed')
        tokens = {r['record_id']: r for r in pq.read_table(token_path).to_pylist()}
        for row in pq.read_table(path).to_pylist():
            refs = json.loads(row['source_refs_json']); messages = json.loads(row['messages_json'])
            memory = json.loads(row['memory_json']); target = messages[-1]['content']; token = tokens.pop(row['record_id'])
            if families.setdefault(row['family_id'], row['split']) != row['split']: raise ValueError('Family split leak')
            if refs['answer_selection'] != {'method':'only_eligible_answer','competitors':1,'ppl_measured':False}:
                raise ValueError('Singleton status')
            if any(row[k] is not None for k in row if k.startswith('mini_') or k.startswith('minicpm_')):
                raise ValueError('Unexpected MiniCPM field')
            if refs['verification_tier'] == 'exact_source_article_extract':
                key = (refs['law_name'], refs['article']); original = originals[key]
                if target != f'《{key[0]}》{key[1]}（所附版本原文）：\n{original}': raise ValueError('Original article mismatch')
                if not any(m['text'].endswith('條文原文：\n' + original) and m['source'] == refs['source_url'] for m in memory):
                    raise ValueError('Original evidence missing')
                if '只做文件摘錄，不判定目前個案能否適用' not in messages[-2]['content']: raise ValueError('Applicability boundary missing')
                for m in memory:
                    name = re.search(r'^法規名稱：([^\n]+)', m['text'])[1]
                    family = base.sha('family:taiwan-civic-v1:' + name)
                    split = 'validation' if int(family[:12],16)%100 < 2 else 'train'
                    if split != row['split']: raise ValueError('Distractor law crosses split')
                counts['exact_original_article_matches'] += 1
            elif refs['verification_tier'] == 'authored_normative_case_not_external_fact':
                if memory[0]['source'] != 'authored:OpenFormosa/taiwan-civic-policy-v1': raise ValueError('Norm disguised as external fact')
                counts['authored_normative_cases_inspected'] += 1
                records.append({'record_id':row['record_id'], 'question':refs['original_query'], 'answer':target,
                    'review_scope':'Owner-authored normative case; no external factual correctness certification'})
            elif refs['verification_tier'] == 'exact_attributed_historical_official_quote':
                quote = re.search('「(.*?)」',target)[1]
                if quote not in memory[0]['text'] or '外交部' not in target or '2022' not in refs['statement_date']:
                    raise ValueError('Official attribution or exact quote')
                counts['exact_official_quote_matches'] += 1
            else: raise ValueError('Unknown source verification tier')
            text,spans=pangolin_render(messages)
            ids,labels=tokenize_spans(tokenizer,text,spans)
            if text!=row['text'] or ids!=token['input_ids'] or labels!=token['labels']:raise ValueError('Full rendering/IDs/masks mismatch')
            request=prepare_dataset_request(row,model='v26',mode='barbet_completion')
            if request['prompt']!=text[:spans[0][0]] or target in request['prompt'] and refs['verification_tier']=='exact_attributed_historical_official_quote':
                raise ValueError('Causal target prefix mismatch')
            if len(ids)+4096>131072 or row['pangolin_tokens']!=len(ids):raise ValueError('Full length/budget mismatch')
            if token['labels'][ids.index(114690)] != 114690 or 114691 in ids:raise ValueError('EOS/PAD loss mask')
            counts['all_records_exact_render_and_full_arrays']+=1
        if tokens:raise ValueError('Token-only orphan')
    if counts['all_records_exact_render_and_full_arrays']!=summary['episodes']:raise ValueError('Coverage mismatch')
    prior.save(output,{'counts':dict(counts),'all_checks_passed':True,'new_cohort_sha256':prior.sha(STATE/'new-cohort-summary.json'),
        'source_lock_sha256':prior.sha(STATE/'source-lock.json'),'law_family_splits_disjoint':True,
        'source_extraction_not_world_truth_or_legal_applicability_verification':True,
        'model_generations_or_training':0, 'authored_normative_case_review':records})
    print(json.dumps({'all_checks_passed':True,'counts':dict(counts)}),flush=True)


if __name__=='__main__':main()
