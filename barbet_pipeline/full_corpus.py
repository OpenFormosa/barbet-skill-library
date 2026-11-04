"""Exhaustive record/array reprocessing, not a sampled inventory.

Every active parent row gets a disposition. Hash-bound targets without an
independent semantic verifier stay in reference_only, outside accepted configs.
No source code executes, no teacher verdict is fabricated, no training starts.
"""
from collections import Counter,defaultdict
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
import copy,json,os,time
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from .common import canonical,digest,read_json,save_new,sha_file
from .corpus import inventory_jobs
from .full_quality import check_content,QualityHold
from barbet_inference.sft import prepare_tools,normalize_messages,pangolin_render,tokenize_spans,portable_messages,Rejected

WORK=Path('/root/barbet-context-sft-data')
STATE=WORK/'reports/full-corpus-v4'
OUTPUT=WORK/'exports/full-corpus-v4'
REPO=Path(__file__).resolve().parents[1]
TOKENIZER_PATH=Path('/root/.cache/huggingface/hub/models--OpenFormosa--barbet-1b-base/snapshots/004ef70881dcb87f1b423951bd418f1748db8b13/tokenizer.json')
QWEN_PATH=WORK/'inference/qwen38/model-fp8'
MAX_TOTAL=131072
RESERVE=4096
PANGOLIN=QWEN=POLICY=READING=None
BASE_STRINGS=['record_id','family_id','split','domain','source','upstream_source','license','verification_tier',
 'observation_origin','memory_placement','query_hash','messages_json','tools_json','memory_json','source_refs_json',
 'skills_json','text','target_spans_json','renderer_version']
EXTRA_STRINGS=['schema_version','parent_record_id','root_task_id','processing_status','verification_scope',
 'checks_json','execution_receipts_json','inference_decisions_json','target_selection_scope','parent_record_sha256']
INTS=['assistant_turns','tool_observations','pangolin_tokens','pangolin_supervised_tokens','output_reserve_tokens']
BOOLS=['actual_retrieval','recorded_error_followup','input_view_changed','natural_answer_rewritten','skill_prefill_updated','semantic_correctness_independently_certified']
RECORD_SCHEMA=pa.schema([(k,pa.string()) for k in BASE_STRINGS+EXTRA_STRINGS]+
 [(k,pa.int64()) for k in INTS]+[(k,pa.bool_()) for k in BOOLS])
TOKEN_SCHEMA=pa.schema([('record_id',pa.string()),('root_task_id',pa.string()),('split',pa.string()),
 ('input_ids',pa.list_(pa.int32())),('labels',pa.list_(pa.int32())),('pangolin_tokens',pa.int64()),('pangolin_supervised_tokens',pa.int64())])
DECISION_SCHEMA=pa.schema([(k,pa.string()) for k in ['parent_record_id','record_id','query_hash','root_task_id','source','split','status','scope','reason']])

def init_worker():
    global PANGOLIN,QWEN,POLICY,READING
    os.environ['TOKENIZERS_PARALLELISM']='false'
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    from transformers import AutoTokenizer,logging
    logging.set_verbosity_error()
    PANGOLIN=Tokenizer.from_file(str(TOKENIZER_PATH))
    QWEN=AutoTokenizer.from_pretrained(str(QWEN_PATH),local_files_only=True,trust_remote_code=False)
    POLICY=read_json(REPO/'configs/source_tool_contract_holds_v1.json')
    # Only source-gold natural-reading replacements already accepted in v3.
    READING={}
    for f in read_json(WORK/'reports/semantic-curriculum-v3/reading-natural-v3/summary.json')['files']:
        if '/records/' not in f['path']:continue
        for row in pq.read_table(f['path'],columns=['record_id','messages_json','tools_json','text','target_spans_json','source_refs_json','public_episode_json']).to_pylist():
            parent=json.loads(row['source_refs_json'])['parent_record_id']
            READING[parent]=row

@lru_cache(maxsize=1024)
def tool_contract(text):return prepare_tools(json.loads(text))

@lru_cache(maxsize=128)
def current_skills(names):
    from .factory import skill_library
    library=skill_library();text,_=library.render(names)
    return text,[{'name':n,'version':library.get(n)['version'],'source_sha256':library.get(n)['source_sha256']} for n in names]

def refresh_singleton_skills(row,messages):
    selection=json.loads(row['source_refs_json']).get('answer_selection',{})
    if selection.get('competitors')!=1:return messages,None
    old=json.loads(row['skills_json']);names=tuple(x if isinstance(x,str) else x['name'] for x in old)
    if not names:return messages,None
    if not messages or messages[0]['role']!='system' or '\n\nAPPLICABLE SKILLS:\n' not in messages[0]['content']:
        raise QualityHold('cannot_locate_explicit_skill_prefill_boundary')
    text,metadata=current_skills(names)
    original=messages[0]['content'];prefix=original.split('\n\nAPPLICABLE SKILLS:\n',1)[0]
    extra=''
    if '\n明示價值設定' in original:extra='\n明示價值設定'+original.split('\n明示價值設定',1)[1]
    updated=copy.deepcopy(messages);updated[0]['content']=prefix+'\n\nAPPLICABLE SKILLS:\n'+text+extra
    return updated,metadata

def transform(row,old_token):
    messages=json.loads(row['messages_json']);tools,validators=tool_contract(row['tools_json'])
    if normalize_messages(messages,tools,validators)!=messages or tools!=json.loads(row['tools_json']):
        raise QualityHold('typed_dialogue_not_identical_to_canonical_normalization')
    text,spans=pangolin_render(messages,tools)
    if text!=row['text'] or [list(s) for s in spans]!=json.loads(row['target_spans_json']):
        raise QualityHold('parent_text_or_spans_mismatch')
    ids,labels=tokenize_spans(PANGOLIN,text,spans)
    if (row['record_id']!=old_token['record_id'] or ids!=old_token['input_ids'] or labels!=old_token['labels']
        or len(ids)!=row['pangolin_tokens'] or sum(t!=-100 for t in labels)!=row['pangolin_supervised_tokens']):
        raise QualityHold('parent_token_arrays_or_counts_mismatch')
    status,scope,execution=check_content(row,messages,tools,POLICY)
    changed=False;refreshed=False;skill_metadata=None
    if row['source']=='rajpurkar/squad_v2':
        if row['record_id'] not in READING:raise QualityHold('v3_first_natural_reading_view_hold_preserved')
        prior=READING[row['record_id']]
        if json.loads(row['source_refs_json'])['answer_selection']['competitors']!=1:
            raise ValueError('Cannot change a ranked multi-candidate input')
        messages=json.loads(prior['messages_json']);tools=json.loads(prior['tools_json'])
        text,spans=pangolin_render(messages,tools);ids,labels=tokenize_spans(PANGOLIN,text,spans)
        if text!=prior['text']:raise ValueError('Previously accepted natural view changed')
        changed=True
    else:
        messages,skill_metadata=refresh_singleton_skills(row,messages)
        refreshed=skill_metadata is not None
        if refreshed:
            text,spans=pangolin_render(messages,tools);ids,labels=tokenize_spans(PANGOLIN,text,spans)
    if not ids or labels[0]!=-100 or len(ids)>MAX_TOTAL or 114691 in ids:
        raise QualityHold('invalid_total_context_or_PAD')
    if any(not text[l:r].endswith('</s>') or messages[t]['role']!='assistant' or not messages[t]['loss'] for l,r,t in spans):
        raise QualityHold('assistant_EOS_or_mask_contract')
    decisions=[]
    for left,right,turn in spans:
        prefix=messages[:turn];prefix_text,_=pangolin_render(prefix,tools);prefix_text+='<|assistant_channel|>'
        if prefix_text!=text[:left]:raise QualityHold('future_or_answer_in_decision_prefix')
        portable=portable_messages(prefix,tools)
        for m in portable:
            m.pop('loss',None)
            if m['role']=='tool':m['role']='user';m['content']='TOOL_OBSERVATION:\n'+m['content']
        chat_text=QWEN.apply_chat_template(portable,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        chat_count=len(QWEN.encode(chat_text,add_special_tokens=False))
        completion_count=len(QWEN.encode(prefix_text,add_special_tokens=False))
        native_count=len(PANGOLIN.encode(prefix_text,add_special_tokens=False).ids)
        if max(chat_count,completion_count,native_count)+RESERVE>MAX_TOTAL:raise QualityHold('complete_inference_budget_exceeded_no_truncation')
        decisions.append({'target_turn':turn,'chat_sha256':digest(portable),'completion_sha256':digest(prefix_text),
            'native_prompt_tokens':native_count,'served_chat_tokens':chat_count,'served_completion_tokens':completion_count,
            'output_reserve_tokens':RESERVE,'total_budget':MAX_TOTAL,'feedback_role':'explicit_user_TOOL_OBSERVATION'})
    rid=digest(['full-corpus-v4',row['record_id'],digest(text)])
    out={k:row.get(k) for k in BASE_STRINGS+INTS+BOOLS}
    out.update(record_id=rid,parent_record_id=row['record_id'],root_task_id=row['family_id'],
        schema_version='barbet-full-reprocess-v4',processing_status=status,verification_scope=scope,
        messages_json=canonical(messages),tools_json=canonical(tools),text=text,target_spans_json=canonical(spans),
        pangolin_tokens=len(ids),pangolin_supervised_tokens=sum(t!=-100 for t in labels),output_reserve_tokens=RESERVE,
        input_view_changed=changed or refreshed,natural_answer_rewritten=changed,skill_prefill_updated=refreshed,
        execution_receipts_json=canonical(execution),inference_decisions_json=canonical(decisions),
        parent_record_sha256=digest(row),semantic_correctness_independently_certified=(scope=='independent_finite_graph_query_and_tool_reexecution'),
        target_selection_scope='same sole eligible target under refreshed input; no alternate scored or PPL claimed' if changed or refreshed else 'exact parent input and selected target retained; frozen v26 decision provenance unchanged',
        checks_json=canonical({'all_parent_arrays_recomputed':True,'all_assistant_prefixes_checked':True,
            'source_and_value_holds_preserved':True,'tools_structurally_valid':True,'scope':scope,
            'learning':'NOT_RUN','transfer':'NOT_RUN','retention':'NOT_RUN','source_programs_executed':False}))
    if refreshed:out['skills_json']=canonical(skill_metadata)
    if changed:
        public=json.loads(prior['public_episode_json']);source_refs=json.loads(row['source_refs_json'])
        source_refs['full_v4_replacement']={'previously_accepted_natural_view_id':prior['record_id'],
            'singleton_source_gold_preserved':True,'input_changed_no_old_PPL_claim':True}
        out.update(actual_retrieval=False,recorded_error_followup=False,assistant_turns=1,tool_observations=0,
            memory_placement=public['memory_placement'],source_refs_json=canonical(source_refs),
            memory_json=canonical({'documents':public['context']}),
            skills_json=canonical([{'name':s['id'],'version':s['version'],'view_sha256':digest(s['text'])} for s in public['skills']]))
    token={'record_id':rid,'root_task_id':row['family_id'],'split':row['split'],'input_ids':ids,'labels':labels,
           'pangolin_tokens':len(ids),'pangolin_supervised_tokens':out['pangolin_supervised_tokens']}
    return out,token,decisions

def process_job(job):
    started=time.time();name=digest(job['path']);receipt=STATE/'receipts'/(name+'.json')
    if receipt.exists():return read_json(receipt)
    if sha_file(job['local'])!=job['sha256'] or sha_file(job['token_local'])!=job['token_sha256']:
        raise ValueError('Parent shard bytes changed')
    excluded=set(job['excluded_indices']);counts=Counter();by_source=Counter();writers={};files=[];decisions=[];offset=0
    def writer(kind,status,split):
        key=(kind,status,split)
        if key not in writers:
            path=OUTPUT/status/kind/(split+'-'+name+'.parquet');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError('Unreceipted output: '+str(path))
            schema=RECORD_SCHEMA if kind=='records' else TOKEN_SCHEMA
            writers[key]=(path,pq.ParquetWriter(path,schema,compression='zstd'))
        return writers[key][1]
    batches=pq.ParquetFile(job['local']).iter_batches(batch_size=32)
    token_batches=pq.ParquetFile(job['token_local']).iter_batches(batch_size=32)
    try:
        for rb,tb in zip_longest(batches,token_batches):
            if rb is None or tb is None or len(rb)!=len(tb):raise ValueError('Parent shards misaligned')
            outputs=defaultdict(list);tokens=defaultdict(list)
            for index,(row,old_token) in enumerate(zip(rb.to_pylist(),tb.to_pylist()),offset):
                if index in excluded:counts['prior_holds_excluded']+=1;continue
                counts['active_rows_processed']+=1;by_source['processed:'+row['source']]+=1
                disposition={'parent_record_id':row['record_id'],'record_id':None,'query_hash':row['query_hash'],
                    'root_task_id':row['family_id'],'source':row['source'],'split':row['split'],'status':None,'scope':None,'reason':None}
                try:
                    out,tok,proof=transform(row,old_token)
                    key=(out['processing_status'],row['split']);outputs[key].append(out);tokens[key].append(tok)
                    disposition.update(record_id=out['record_id'],status=out['processing_status'],scope=out['verification_scope'])
                    counts['exported']+=1;counts['status:'+out['processing_status']]+=1;counts['split:'+row['split']]+=1
                    counts['input_tokens']+=out['pangolin_tokens'];counts['supervised_tokens']+=out['pangolin_supervised_tokens']
                    counts['decision_points']+=len(proof);counts['tool_reexecutions']+=len(json.loads(out['execution_receipts_json']))
                    counts['natural_views_replaced']+=out['natural_answer_rewritten'];counts['skill_prefills_updated']+=out['skill_prefill_updated']
                    by_source[out['processing_status']+':'+row['source']]+=1
                except (QualityHold,Rejected) as error:
                    disposition.update(status='quarantined',reason=str(error));counts['quarantined']+=1
                    counts['hold:'+str(error)]+=1;by_source['quarantined:'+row['source']]+=1
                decisions.append(disposition)
            offset+=len(rb)
            for (status,split),values in outputs.items():
                writer('records',status,split).write_table(pa.Table.from_pylist(values,schema=RECORD_SCHEMA))
                writer('tokenized',status,split).write_table(pa.Table.from_pylist(tokens[(status,split)],schema=TOKEN_SCHEMA))
    finally:
        for path,w in writers.values():w.close()
    if counts['prior_holds_excluded']!=len(excluded):raise ValueError('Prior hold coverage differs')
    for key,(path,w) in writers.items():
        kind,status,split=key
        if kind=='records':
            token_path=OUTPUT/status/'tokenized'/path.name
            for a,b in zip_longest(pq.ParquetFile(path).iter_batches(batch_size=64),pq.ParquetFile(token_path).iter_batches(batch_size=64)):
                if a is None or b is None or len(a)!=len(b):raise ValueError('Written arrays differ')
                for r,t in zip(a.to_pylist(),b.to_pylist()):
                    if any(r[k]!=t[k] for k in ['record_id','root_task_id','split','pangolin_tokens','pangolin_supervised_tokens']):raise ValueError('Saved token identities differ')
                    ids,labels=tokenize_spans(PANGOLIN,r['text'],json.loads(r['target_spans_json']))
                    if ids!=t['input_ids'] or labels!=t['labels']:raise ValueError('Saved arrays fail independent reread')
        files.append({'local':str(path),'remote':'full-v4/'+status+'/'+kind+'/'+path.name,
                      'sha256':sha_file(path),'bytes':path.stat().st_size,'rows':pq.ParquetFile(path).metadata.num_rows})
    path=OUTPUT/'dispositions'/(name+'.parquet');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    pq.write_table(pa.Table.from_pylist(decisions,schema=DECISION_SCHEMA),path,compression='zstd')
    files.append({'local':str(path),'remote':'full-v4/dispositions/'+path.name,'sha256':sha_file(path),'bytes':path.stat().st_size,'rows':len(decisions)})
    result={'source_path':job['path'],'source_sha256':job['sha256'],'token_source_sha256':job['token_sha256'],
        'counts':dict(counts),'by_source':dict(by_source),'files':files,'seconds':time.time()-started,
        'driver_sha256':sha_file(Path(__file__)),'quality_driver_sha256':sha_file(REPO/'barbet_pipeline/full_quality.py')}
    save_new(receipt,result)
    return result

def prepare():
    split=read_json(STATE/'split-preflight.json')
    if any(split['conflicts'].values()):raise ValueError('Resolve actual split/identity conflicts before export')
    old=read_json(WORK/'reports/memory-harness-v2-barbet-readiness-v1/plan.json')['base_manifest']
    old+=read_json(WORK/'reports/taiwan-civic-v1/new-cohort-summary.json')['files']
    by_path={i['path']:i for i in old};jobs=inventory_jobs(WORK)
    for job in jobs:
        token_path=job['path'].replace('/records/','/tokenized/')
        job['token_local']=job['local'].replace('/records/','/tokenized/')
        job['token_sha256']=by_path[token_path]['sha256']
    plan={'jobs':jobs,'active_rows_required':10018967,'mode':'FULL_CORPUS_NO_SAMPLE_LIMIT',
        'parent_dataset':'b88f4ff7171bb6fb8c1b775ae21de814ea2123fb','preserve_prior_holds':True,
        'no_model_training':True,'all_rows_recompute_arrays_and_prefixes':True,
        'source_reference_is_not_independent_semantic_certification':True,
        'reference_only_excluded_from_accepted_training_configs':True,
        'tokenizer_sha256':sha_file(TOKENIZER_PATH),'served_tokenizer_sha256':sha_file(QWEN_PATH/'tokenizer.json'),
        'total_budget':MAX_TOTAL,'output_reserve':RESERVE,'driver_sha256':sha_file(Path(__file__)),
        'quality_driver_sha256':sha_file(REPO/'barbet_pipeline/full_quality.py')}
    save_new(STATE/'plan.json',plan);return plan
