"""Versioned learner records and exact Pangolin arrays; no model training."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path

from .common import canonical, digest, sha_file, read_json, save_new
from .visibility import visible_prefix
from barbet_inference.sft import prepare_tools, normalize_messages, pangolin_render, tokenize_spans, portable_messages


TOKENIZER_SHA = 'f84754e42cc4b08d5573a6b7306635e86ed04fff68ee707662d76d7044917176'


def load_tokenizer(path):
    from tokenizers import Tokenizer
    if sha_file(path) != TOKENIZER_SHA:
        raise ValueError('Pangolin tokenizer identity changed')
    return Tokenizer.from_file(str(path))


def learner_row(*, tokenizer, metadata, public, messages, receipts, source_refs, max_total=8192, reserve=4096):
    tools, validators = prepare_tools(public['tools'])
    messages = normalize_messages(messages,tools,validators)
    text, spans = pangolin_render(messages,tools)
    ids, labels = tokenize_spans(tokenizer,text,spans)
    if len(ids) > max_total or len(ids)!=len(labels) or labels[0]!=-100:
        raise ValueError('Invalid complete sequence budget or unshifted labels')
    if not spans or any(messages[n]['role']!='assistant' or not messages[n]['loss'] for _,_,n in spans):
        raise ValueError('Only accepted assistant turns may receive loss')
    # Independently check literal boundaries and EOS on each supervised span.
    if any(not text[left:right].endswith('</s>') for left,right,_ in spans):
        raise ValueError('Missing supervised assistant EOS')
    starts=[i for i,m in enumerate(messages) if m['role']=='assistant' and m['loss']]
    # Each decision prefix ends before its target, with no future observations.
    decisions=[]
    for i in starts:
        prefix=visible_prefix(messages,i)
        rendered,_=pangolin_render(prefix,tools)
        rendered+='<|assistant_channel|>'
        prompt_ids=tokenizer.encode(rendered,add_special_tokens=False).ids
        if len(prompt_ids)+reserve>max_total:
            raise ValueError('Decision prompt plus reserved output exceeds budget')
        chat=portable_messages(prefix,tools)
        # Portable Chat uses explicit user continuations for tool observations,
        # matching the Pangolin renderer instead of switching feedback roles.
        for m in chat:
            m.pop('loss',None)
            if m['role']=='tool':
                m['role']='user';m['content']='TOOL_OBSERVATION:\n'+m['content']
        decisions.append({'target_turn':i,'chat':chat,'completion_prompt':rendered,
            'chat_sha256':digest(chat),'completion_sha256':digest(rendered),
            'pangolin_prompt_tokens':len(prompt_ids),'output_reserve_tokens':reserve})
    row={**metadata,'schema_version':'barbet-semantic-learner-v3',
         'public_episode_json':canonical(public),'messages_json':canonical(messages),
         'tools_json':canonical(tools),'execution_receipts_json':canonical(receipts),
         'source_refs_json':canonical(source_refs),'text':text,
         'target_spans_json':canonical(spans),'inference_decisions_json':canonical(decisions),
         'pangolin_tokens':len(ids),'pangolin_supervised_tokens':sum(x!=-100 for x in labels),
         'renderer_version':'barbet_portable_v1', 'tokenizer_sha256':TOKENIZER_SHA,
         'target_selection':'independently_eligible_deterministic_singleton_unscored',
         'v26_ppl':None,'learning_status':'NOT_RUN','transfer_status':'NOT_RUN','retention_status':'NOT_RUN',
         'training_admission':'CONTENT_CHECKED_EXPERIMENTAL_REQUIRES_PILOT',
         'output_reserve_tokens':reserve}
    token={'record_id':metadata['record_id'],'root_task_id':metadata['root_task_id'],
           'split':metadata['split'],'input_ids':ids,'labels':labels,
           'pangolin_tokens':len(ids),'pangolin_supervised_tokens':row['pangolin_supervised_tokens']}
    return row,token


def write_shard(output, name, records, tokens):
    import pyarrow as pa
    import pyarrow.parquet as pq
    output=Path(output)
    if not records or len(records)!=len(tokens):
        raise ValueError('Empty or mismatched learner shard')
    out=[]
    for config,rows in [('records',records),('tokenized',tokens)]:
        path=output/config/(name+'.parquet');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise FileExistsError(path)
        if config=='tokenized':
            schema=pa.schema([('record_id',pa.string()),('root_task_id',pa.string()),('split',pa.string()),
                ('input_ids',pa.list_(pa.int32())),('labels',pa.list_(pa.int32())),
                ('pangolin_tokens',pa.int64()),('pangolin_supervised_tokens',pa.int64())])
            table=pa.Table.from_pylist(rows,schema=schema)
        else: table=pa.Table.from_pylist(rows)
        pq.write_table(table,path,compression='zstd')
        if not pq.read_table(path).equals(table):
            raise ValueError('Learner shard roundtrip differs')
        out.append({'path':str(path),'sha256':sha_file(path),'bytes':path.stat().st_size,'rows':len(rows)})
    return out


def validate_shard(records_path,tokens_path,tokenizer):
    import pyarrow.parquet as pq
    records=pq.read_table(records_path).to_pylist()
    tokens=pq.read_table(tokens_path).to_pylist()
    if len(records)!=len(tokens):raise ValueError('Shard length mismatch')
    counts=Counter()
    for r,t in zip(records,tokens):
        if any(r[k]!=t[k] for k in ['record_id','root_task_id','split']):raise ValueError('Row identities differ')
        messages=json.loads(r['messages_json']);tools=json.loads(r['tools_json'])
        text,spans=pangolin_render(messages,tools)
        ids,labels=tokenize_spans(tokenizer,text,spans)
        if text!=r['text'] or ids!=t['input_ids'] or labels!=t['labels']:
            raise ValueError('Renderer or exact labels differ')
        if labels[0]!=-100 or len(ids)>8192:
            raise ValueError('Sequence contract differs')
        for d in json.loads(r['inference_decisions_json']):
            prefix=visible_prefix(messages,d['target_turn']);p,_=pangolin_render(prefix,tools)
            if p+'<|assistant_channel|>'!=d['completion_prompt']:
                raise ValueError('Future observation or target leaked into decision prefix')
        counts['rows']+=1;counts['input_tokens']+=len(ids);counts['supervised_tokens']+=sum(x!=-100 for x in labels)
    return dict(counts)
