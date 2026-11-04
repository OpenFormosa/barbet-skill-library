"""Reprocess admitted SQuAD source rows into natural, source-only learner views.

Only rows present in the audited current release are eligible. Existing article
splits and first holds are preserved. This is source-gold span verification, not
an independent proof of world truth, difficulty, transfer or retention.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path

from .common import canonical,digest,sha_file,read_json,save_new
from .corpus import inventory_jobs
from .export import learner_row,write_shard,load_tokenizer
from .factory import library_card
from .visibility import actor_messages


def reprocess_reading(work, output, tokenizer_path):
    import pyarrow.parquet as pq
    work,output=Path(work),Path(output)
    if (output/'summary.json').exists():raise FileExistsError('Reading export already sealed')
    lock=read_json(work/'reports/memory-harness-v2/build-source-lock.json')['files']
    source=next(x for x in lock if x['repo']=='rajpurkar/squad_v2' and x['path'].endswith('.parquet'))
    if sha_file(source['local'])!=source['sha256']:raise ValueError('Natural source hash changed')
    raw=pq.read_table(source['local']).to_pylist()
    receipts=[read_json(p) for p in (work/'reports/semantic-curriculum-v3/corpus/receipts').glob('*.json')]
    wanted={r['source_path']:r for r in receipts if r['counts'].get('source:rajpurkar/squad_v2',0)}
    jobs=[j for j in inventory_jobs(work) if j['path'] in wanted]
    plan={'source':source,'jobs':[{'path':j['path'],'sha256':j['sha256'],'prior_holds':len(j['excluded_indices'])} for j in jobs],
          'scope':'source_gold_answer_and_exact_span; inherited article grouping and value holds',
          'targets':'source canonical gold, natural answer without forced JSON',
          'skill_view':'matched source task with or without pinned evidence skill, deterministic by root',
          'driver_sha256':sha_file(Path(__file__)),'exporter_sha256':sha_file(Path(__file__).with_name('export.py')),
          'expected_rows':114126,'new_independent_natural_questions':0,
          'identity_version':'pinned_upstream_task_id_v2_not_question_hash',
          'prior_failed_attempt':'reading-natural/first-failure.json'}
    save_new(output/'plan.json',plan)
    tokenizer=load_tokenizer(tokenizer_path)
    counts=Counter(); files=[]; buffers={}; indices=Counter(); seen=set(); title_splits={}; exclusions=[]
    accounted=0
    columns=['record_id','family_id','query_hash','split','source','source_refs_json']
    for job in jobs:
        if sha_file(job['local'])!=job['sha256']:raise ValueError('Original release input hash changed')
        excluded=set(job['excluded_indices']);row_index=0
        for batch in pq.ParquetFile(job['local']).iter_batches(batch_size=512,columns=columns):
            for old in batch.to_pylist():
                index=row_index;row_index+=1
                if index in excluded or old['source']!='rajpurkar/squad_v2':continue
                refs=json.loads(old['source_refs_json']);q=raw[refs['row_index']]
                if refs.get('file_sha256')!=source['sha256'] or refs.get('upstream_id')!=q['id']:
                    raise ValueError('Row provenance mismatch')
                answers=q['answers'];answer=answers['text'][0] if answers['text'] else None
                if len(answers['text'])!=len(answers['answer_start']):raise ValueError('Gold alignment mismatch')
                for a,s in zip(answers['text'],answers['answer_start']):
                    if not a or q['context'][s:s+len(a)]!=a:raise ValueError('Gold answer span mismatch')
                rid=old['record_id'];root=digest(['rajpurkar/squad_v2',source['revision'],q['id']]);split=old['split']
                accounted+=1
                if root in seen:
                    counts['duplicate_source_tasks_collapsed']+=1
                    continue
                seen.add(root)
                title=q['title']
                if title in title_splits and title_splits[title]!=split:raise ValueError('Inherited article split overlaps')
                title_splits[title]=split
                with_skill=int(root[:2],16)%2==0
                public={'task':q['question']+'\nAnswer in English using only the supplied passage. If it does not support an answer, state that the passage is insufficient.',
                    'context':[{'id':'source-passage','title':title,'revision':source['revision'],'text':q['context']}],
                    'skills':[library_card('grounded-evidence-answer')] if with_skill else [],'tools':[]}
                messages=actor_messages(public)
                messages.append({'role':'assistant','content':answer if answer is not None else 'The supplied passage does not provide enough information to answer this question.','loss':True})
                metadata={'record_id':digest(['reading-natural-v3',rid]),'root_task_id':root,
                    'source_group_id':old['family_id'],'semantic_family_id':'source_only_reading',
                    'program_graph_id':'answer_from_visible_passage' if answer is not None else 'recognize_unsupported_question',
                    'surface_template_id':'upstream_natural_question_'+digest(q['question']),
                    'counterfactual_group_id':root,'variant':'natural_answer_with_skill' if with_skill else 'natural_answer_bare',
                    'split':split,'source':'rajpurkar/squad_v2','license':'cc-by-sa-4.0',
                    'verification_scope':'upstream_gold_label_and_exact_answer_span_NOT_world_truth_or_learnability'}
                provenance={'parent_dataset_revision':'b88f4ff7171bb6fb8c1b775ae21de814ea2123fb',
                    'parent_record_id':rid,'parent_query_hash':old['query_hash'],'source':refs,
                    'retained_after_all_prior_holds':True,'article_split_preserved':True,
                    'change':'remove forced JSON/tool wrapper; source question, passage and canonical answer preserved',
                    'skill_provenance':'bundled versioned library' if with_skill else 'none'}
                try:
                    r,t=learner_row(tokenizer=tokenizer,metadata=metadata,public=public,messages=messages,receipts=[],source_refs=provenance)
                except ValueError as exc:
                    exclusions.append({'record_id':rid,'reason':str(exc)});continue
                rr,tt=buffers.setdefault(split,([],[]));rr.append(r);tt.append(t)
                counts['rows']+=1;counts['split:'+split]+=1;counts['skill_view:'+str(with_skill)]+=1
                counts['answerable:'+str(answer is not None)]+=1
                counts['input_tokens']+=r['pangolin_tokens'];counts['supervised_tokens']+=r['pangolin_supervised_tokens']
                if len(rr)>=1024:
                    files+=write_shard(output,f'{split}-{indices[split]:05d}',rr,tt);indices[split]+=1;buffers[split]=([],[])
        print(json.dumps({'reading_rows':counts['rows'],'source_shard':job['path']}),flush=True)
    for split,(rr,tt) in buffers.items():
        if rr:files+=write_shard(output,f'{split}-{indices[split]:05d}',rr,tt)
    if accounted!=plan['expected_rows']:raise ValueError('Not all admitted reading rows accounted for')
    result={'counts':dict(counts),'source_rows_accounted_for':accounted,'unique_source_task_ids':len(seen),'holds':exclusions,
            'source_articles':len(title_splits),'files':files,'general_capability_retention':'NOT_RUN',
            'not_a_complete_general_retention_mixture':True,'new_root_tasks':0,
            'unique_existing_natural_questions_reprocessed':len(seen)-len(exclusions)}
    save_new(output/'summary.json',result)
    return result
