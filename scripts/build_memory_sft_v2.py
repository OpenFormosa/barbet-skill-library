"""Reproducible source -> causal records -> exact token views. No model training.

Run in the unchanged data environment, with the repository and the system's pinned
Jinja2 on PYTHONPATH. Source programs are always inert strings.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

from barbet_inference.library import SkillLibrary
from barbet_inference.memory import MemoryItem, compose_memory_prompt, canonical
from barbet_inference.sft import (MiniCPMRenderer, Rejected, augment_system, clean_text,
    inference_messages, normalize_messages, pangolin_render, portable_messages,
    prepare_tools, tokenize_spans)
from testtime.retrieval import Document, LexicalIndex

WORK = Path('/root/barbet-context-sft-data')
REPO = Path(__file__).resolve().parents[1]
RAW = WORK/'data/raw_memory_harness_v2'
OUT = WORK/'exports/memory-harness-v2'
STATE = WORK/'reports/memory-harness-v2'
ALLOWED = {'mit', 'apache-2.0', 'cc-by-4.0', 'cc-by-sa-4.0'}
ROUTES = {
    'reading': ['grounded-evidence-answer', 'unsupported-question-calibration'],
    'math': ['arithmetic-modeling', 'unit-conversion'],
    'coding': ['code-contract-check', 'execution-feedback-repair'],
    'tool_use': ['tool-selection-and-arguments', 'external-state-grounding'],
    'search': ['retrieval-query-refinement', 'grounded-evidence-answer'],
    'web': ['external-state-grounding', 'tool-selection-and-arguments'],
}


def sha(value):
    return hashlib.sha256((value if isinstance(value, bytes) else value.encode())).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def norm(text, numbers=False):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'\s+', ' ', text).strip()
    return re.sub(r'\d+(?:[.,]\d+)*', '#', text) if numbers else text


def words(text):
    return re.findall(r'[\w]+', norm(text))


def ngrams(text, n=13):
    w = words(text)
    return [' '.join(w[i:i+n]) for i in range(len(w)-n+1)]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def exclusion_inputs():
    texts, prior_refs, receipts = set(), set(), []
    base = WORK/'data/migrations/skill-prefill-v1/plan.json'
    plans = [base, *sorted((WORK/'data/expansion/skill-prefill-v1').glob('*/plan.json')),
             *sorted((WORK/'data/expansion/skill-prefill-v1').glob('*/candidates.json'))]
    for p in plans:
        data = json.loads(p.read_text())
        rows = data if isinstance(data, list) else data.get('items', data.get('records', data.get('candidates', [])))
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict):
                continue
            for k in ('source_question', 'source_instruction', 'source_query', 'query', 'question'):
                if isinstance(r.get(k), str):
                    texts.add(norm(r[k]))
            for ref in r.get('source_refs', []):
                if 'row_index' in ref and 'source_id' in ref:
                    prior_refs.add(f"{ref['source_id']}:{ref['row_index']}")
            if r.get('source_id') and 'row_index' in r:
                prior_refs.add(f"{r['source_id']}:{r['row_index']}")
    benchmark_texts = set()
    files = list(Path('/root/minicpm-skill-study/data').glob('*.inputs.jsonl'))
    files += list(Path('/root/barbet-skill-library/experiments/test_time_v1/work').glob('*.inputs.jsonl'))
    files += list(Path('/root/barbet-qwen-benchmark/runs').glob('20261024-v26*/questions*.jsonl'))
    def extract(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ('question', 'query', 'prompt', 'input') and isinstance(v, str):
                    benchmark_texts.add(norm(v))
                elif k in ('messages', 'inputs', 'input') and isinstance(v, (list, dict)):
                    extract(v)
            if o.get('role') == 'user' and isinstance(o.get('content'), str):
                benchmark_texts.add(norm(o['content']))
        elif isinstance(o, list):
            for v in o:
                extract(v)
    for p in sorted(set(files)):
        receipts.append({'path': str(p), 'sha256': file_sha(p)})
        for line in p.read_text().splitlines():
            if line.strip():
                extract(json.loads(line))
    if not benchmark_texts:
        raise ValueError('missing frozen evaluation inputs')
    return {'old_query_hashes': sorted(sha(t) for t in texts), 'old_source_rows': sorted(prior_refs),
            'benchmark_query_hashes': sorted(sha(t) for t in benchmark_texts),
            'benchmark_13gram_hashes': sorted({sha(g) for t in benchmark_texts for g in ngrams(t)}),
            'benchmark_files': receipts, 'prior_plan_files': len(plans),
            'scope': 'Exact normalized prior queries/row locators, and local benchmark exact/13-word overlap; not universal semantic decontamination.'}


def prepare():
    STATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((RAW/'sources.lock.json').read_text())
    # The MIT Orca source was already downloaded and pinned in the earlier source
    # refresh; original solution text is imported, never regenerated by a teacher.
    orca = WORK/'data/raw/public-expansion-20261027/orca-math-train.parquet'
    lock['files'].append({'repo': 'microsoft/orca-math-word-problems-200k',
        'revision': '29255d1770cc4eac66e5e7fa378cba542c026350',
        'path': 'data/train-00000-of-00001.parquet', 'local': str(orca),
        'bytes': orca.stat().st_size, 'sha256': file_sha(orca)})
    jobs = []
    for source in lock['files']:
        p = Path(source['local'])
        if p.suffix not in {'.jsonl', '.parquet'}:
            continue
        if file_sha(p) != source['sha256']:
            raise ValueError('source bytes changed')
        if p.suffix == '.parquet':
            count = pq.ParquetFile(p).metadata.num_rows
            for start in range(0, count, 2000):
                jobs.append({'source': source, 'start': start, 'stop': min(start+2000, count), 'format': 'parquet'})
        else:
            start_byte, start_row = 0, 0
            with p.open('rb') as f:
                n = 0
                for line in f:
                    n += 1
                    if n % 2000 == 0:
                        end = f.tell()
                        jobs.append({'source': source, 'start': start_row, 'stop': n,
                                     'start_byte': start_byte, 'stop_byte': end, 'format': 'jsonl'})
                        start_byte, start_row = end, n
                if n != start_row:
                    jobs.append({'source': source, 'start': start_row, 'stop': n,
                                 'start_byte': start_byte, 'stop_byte': f.tell(), 'format': 'jsonl'})
    jobs.sort(key=lambda j: (j['source']['repo'], j['source']['path'], j['start']))
    for n, j in enumerate(jobs):
        j['job_id'] = f'{n:05d}'
    save(STATE/'build-source-lock.json', lock)
    save(STATE/'exclusions.private.json', exclusion_inputs())
    save(STATE/'normalize-jobs.json', jobs)
    print(json.dumps({'jobs': len(jobs), 'source_rows': sum(j['stop']-j['start'] for j in jobs)}), flush=True)


def init_normalizer():
    global EX, PRIMARY, LIB, SKILLS
    raw = json.loads((STATE/'exclusions.private.json').read_text())
    EX = {k: set(raw[k]) for k in ('old_query_hashes', 'old_source_rows', 'benchmark_query_hashes', 'benchmark_13gram_hashes')}
    PRIMARY = {r['repo']: r for r in json.loads((RAW/'upstream-notices/lock.json').read_text())}
    LIB = SkillLibrary(REPO)
    SKILLS = {domain: LIB.render(names) for domain, names in ROUTES.items()}


@lru_cache(maxsize=256)
def tool_cache(serialized):
    return prepare_tools(json.loads(serialized))


def qa_task(row, sid):
    query = row['question']
    documents = []
    evidence = []
    if sid == 'rajpurkar/squad_v2':
        text = row['context']
        answers = row['answers']
        if len(answers['text']) != len(answers['answer_start']):
            raise Rejected('squad_answer_alignment')
        for answer, start in zip(answers['text'], answers['answer_start']):
            if not answer or text[start:start+len(answer)] != answer:
                raise Rejected('squad_answer_span')
        answer = answers['text'][0] if answers['text'] else None
        documents = [Document('d0', text, sid, row['title'])]
        evidence = ['d0'] if answer is not None else []
        group = 'squad-article:' + row['title']
    else:
        titles, sentences = row['context']['title'], row['context']['sentences']
        if len(set(titles)) != len(titles):
            raise Rejected('hotpot_duplicate_title')
        for n, (title, strings) in enumerate(zip(titles, sentences)):
            for k, text in enumerate(strings):
                documents.append(Document(f'd{n}s{k}', text, sid, title))
        sf = row['supporting_facts']
        for title, k in zip(sf['title'], sf['sent_id']):
            if title not in titles or not 0 <= k < len(sentences[titles.index(title)]):
                raise Rejected('hotpot_support_outside_context')
            evidence.append(f'd{titles.index(title)}s{k}')
        evidence = list(dict.fromkeys(evidence))
        if not evidence:
            raise Rejected('hotpot_empty_support')
        answer = row['answer']
        group = 'hotpot-pair:' + canonical(sorted(set(sf['title'])))
    target = canonical({'answer': answer, 'evidence_ids': evidence})
    contract = ('\nReturn one JSON object with answer and evidence_ids. Use only the supplied '
                'documents. If the documents do not support an answer, set answer to null '
                'and evidence_ids to []. Evidence IDs must identify supporting documents/spans.')
    placement = 'front' if int(sha(query)[:8],16) % 2 else 'back'
    items = [MemoryItem(d.id, d.title+'\n'+d.text, d.source) for d in documents]
    use_retrieval = int(sha(query)[8:16],16) % 4 == 0
    raw_tools, retrieval = [], False
    messages = []
    if use_retrieval:
        k = min(16, len(documents))
        selected = [d for d, _ in LexicalIndex(documents).search(query, k)]
        selected_ids = {d.id for d in selected}
        # Never attach an answer after retrieval that removed its annotated support.
        if set(evidence) <= selected_ids and (selected or answer is None):
            retrieval = True
            raw_tools = [{'type':'function','function':{'name':'memory_search',
                'description':'Search the provided frozen document collection. Returns source IDs and complete matching spans.',
                'parameters':{'type':'object','properties':{'query':{'type':'string'},'k':{'type':'integer','minimum':1,'maximum':16}},
                              'required':['query','k'],'additionalProperties':False}}}]
            response = [{'id':d.id,'title':d.title,'text':d.text,'source':d.source} for d in selected]
            messages = [{'role':'user','content':query+contract+'\nA document collection is available through memory_search.'},
                {'role':'assistant','content':'','tool_calls':[{'id':'memory_0','type':'function',
                 'function':{'name':'memory_search','arguments':{'query':query,'k':k}}}]},
                {'role':'tool','tool_call_id':'memory_0','name':'memory_search','content':canonical(response)},
                {'role':'assistant','content':target}]
    if not retrieval:
        messages = [{'role':'user','content':compose_memory_prompt(query+contract, items, placement=placement)},
                    {'role':'assistant','content':target}]
    return query, messages, raw_tools, {'documents':[d.__dict__ for d in documents]}, group, placement, retrieval


def normalize_row(row, source, index):
    sid = source['repo']
    origin, license_name, domain, group = sid, None, None, None
    metadata, memory = {}, {}
    placement, retrieval = 'chronological', False
    if sid in {'rajpurkar/squad_v2', 'hotpotqa/hotpot_qa'}:
        query, raw, raw_tools, memory, group, placement, retrieval = qa_task(row, sid)
        license_name, domain = 'cc-by-sa-4.0', 'reading'
        upstream_id = row['id']
        tier = 'source_gold_span_checks' if 'squad' in sid else 'source_gold_support_index_checks'
        observation = 'local_deterministic_retrieval' if retrieval else 'provided_document'
    elif sid == 'microsoft/orca-math-word-problems-200k':
        query = row['question']
        raw = [{'role':'user','content':query},{'role':'assistant','content':row['answer']}]
        raw_tools, license_name, domain = [], 'mit', 'math'
        upstream_id, tier, observation = str(index), 'upstream_solution_not_locally_math_verified', 'user_problem_premises'
    elif sid == 'voidful/agent-sft':
        metadata = json.loads(row['metadata'])
        license_name = metadata.get('license')
        origin = metadata.get('hf_id')
        primary = PRIMARY.get(origin, {})
        if license_name not in ALLOWED or primary.get('primary_license') != license_name:
            raise Rejected('license_unclear_or_primary_mismatch')
        if re.search(r'(^|[/_-])(test|validation|eval)([/_.-]|$)', row['source_subset'], re.I):
            raise Rejected('upstream_evaluation_split')
        quality = json.loads(row['quality'])
        if quality.get('tier') not in {'high', 'medium'}:
            raise Rejected('upstream_low_quality')
        raw, raw_tools = json.loads(row['messages']), json.loads(row['tools'])
        code = row['source'].startswith('swe_') or row['source'] in {'nemotron_terminal','coderforge'}
        domain = 'coding' if code else ('web' if row['source'] in {'nnetnav','mind2web'} else 'tool_use')
        if not raw_tools:
            raise Rejected('no_context_or_action_signal')
        upstream_id = row['id']
        if metadata.get('instance_id'):
            group = 'swe-instance:' + metadata['instance_id']
        elif code and metadata.get('task'):
            group = 'terminal-task:' + metadata['task']
        observation = 'upstream_recorded_environment' if code else 'upstream_recorded_or_simulated'
        tier = 'upstream_verified_success_not_reexecuted' if quality.get('signals',{}).get('verified_success') else 'upstream_target_structure_checked'
        query = next((clean_text(m.get('content')) for m in raw if m.get('role') == 'user'), '')
    else:
        if row.get('filter_reason') or row.get('matched_categories') or row.get('match_contexts'):
            raise Rejected('upstream_filter_or_contamination_flag')
        raw, raw_tools = row['messages'], row.get('tools', [])
        metadata = row.get('metadata') or {}
        upstream_id = row.get('uuid') or metadata.get('uuid') or str(index)
        license_name = 'cc-by-4.0'
        domain = 'search' if source['path'].endswith('/search.jsonl') else 'tool_use'
        observation = 'upstream_web_snapshot' if domain == 'search' else 'upstream_simulated_tool_environment'
        tier = 'upstream_target_structure_checked'
        query = next((clean_text(m.get('content')) for m in raw if m.get('role') == 'user'), '')
    if not query or len(query.strip()) < 12:
        raise Rejected('insufficient_task_identity')
    if f'{sid}:{index}' in EX['old_source_rows'] or sha(norm(query)) in EX['old_query_hashes']:
        raise Rejected('prior_job_or_legacy_source_exclusion')
    if sha(norm(query)) in EX['benchmark_query_hashes'] or any(sha(g) in EX['benchmark_13gram_hashes'] for g in ngrams(query)):
        raise Rejected('local_evaluation_overlap')
    tools, validators = tool_cache(canonical(raw_tools))
    messages = normalize_messages(raw, tools, validators)
    # Reject literal repeated loops, not legitimate failed tests or observations.
    targets = [canonical(m) for m in messages if m['role'] == 'assistant']
    if any(n >= 4 for n in Counter(targets).values()):
        raise Rejected('repeated_assistant_loop')
    group = group or 'query-family:' + norm(query, numbers=True)
    family_id = sha(group)
    # Derive grouping before any skill/memory placement. Root families include
    # conservative numerical templates and source instance IDs, not fake unique IDs.
    split = 'validation' if int(family_id[:12],16) % 100 < 2 else 'train'
    identity = canonical({'messages':messages,'tools':tools})
    dedup_key = sha(identity)
    messages = augment_system(messages, SKILLS[domain][0])
    record_id = sha(canonical({'version':'memory-harness-sft-v2','messages':messages,'tools':tools}))
    provenance = {'repo':sid,'revision':source['revision'],'file':source['path'],'file_sha256':source['sha256'],
        'row_index':index,'upstream_id':str(upstream_id),'upstream_repo':origin,
        'license':license_name,'attribution_url':'https://huggingface.co/datasets/'+origin,
        'raw_row_sha256':sha(canonical(row)), 'new_semantic_generation':False,
        'transformations':['assistant reasoning channel excluded','lossless JSON tool adapter',
                           'causal skill prefill','reserved chat delimiters escaped']}
    # Explicit allowlist: do not publish upstream ground_truth, judge notes,
    # user accounts/paths in processing metadata, or private source job artifacts.
    for key in ('instance_id','repo','task','source_subset'):
        if key in metadata:
            provenance['original_'+key] = metadata[key]
    tool_texts = [m['content'] for m in messages if m['role']=='tool']
    has_error = any(re.search(r'(?i)traceback|\b(error|failed|failure|exception|timeout)\b',t) for t in tool_texts)
    return {'record_id':record_id,'family_id':family_id,'split':split,'domain':domain,
        'source':sid,'upstream_source':origin,'license':license_name,'dedup_key':dedup_key,
        'query_hash':sha(norm(query)), 'messages':messages,'tools':tools,'memory':memory,
        'source_refs':provenance,'skill_ids':ROUTES[domain],'skills':SKILLS[domain][1],
        'verification_tier':tier,'observation_origin':observation,'memory_placement':placement,
        'actual_retrieval':retrieval,'recorded_error_followup':has_error,
        'locally_executed_source_program':False,
        'assistant_turns':sum(m['role']=='assistant' and m['loss'] for m in messages),
        'tool_observations':len(tool_texts)}


def read_job(job):
    path = job['source']['local']
    if job['format'] == 'jsonl':
        with open(path, 'rb') as f:
            f.seek(job['start_byte'])
            for n in range(job['start'], job['stop']):
                yield n, json.loads(f.readline())
    else:
        parquet = pq.ParquetFile(path)
        offset = 0
        for rg in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(rg).num_rows
            if offset+count > job['start'] and offset < job['stop']:
                begin, end = max(0,job['start']-offset), min(count,job['stop']-offset)
                for n, row in enumerate(parquet.read_row_group(rg).slice(begin,end-begin).to_pylist(), offset+begin):
                    yield n, row
            offset += count


def normalize_job(job):
    ident = job['job_id']
    path = STATE/'normalized'/f'{ident}.jsonl'
    report = STATE/'normalized'/f'{ident}.report.json'
    if report.exists():
        return json.loads(report.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with path.open('w') as f, (STATE/'normalized'/f'{ident}.excluded.jsonl').open('w') as reject:
        for index, row in read_job(job):
            counts['seen'] += 1
            try:
                value = normalize_row(row, job['source'], index)
            except Rejected as exc:
                counts['excluded:'+str(exc)] += 1
                reject.write(canonical({'row_index':index,'reason':str(exc)})+'\n')
                continue
            f.write(canonical(value)+'\n')
            counts['accepted'] += 1
    result = {'job_id':ident,'source':job['source']['repo'],'counts':dict(counts),
              'path':str(path),'sha256':file_sha(path)}
    save(report,result)
    return result


def normalize(workers, pilot=False):
    jobs = json.loads((STATE/'normalize-jobs.json').read_text())
    if pilot:
        selected = {}
        for job in jobs:
            key = job['source']['path']
            if key not in selected:
                selected[key] = {**job,'stop':min(job['start']+20,job['stop']), 'job_id':'pilot-'+job['job_id']}
        jobs = list(selected.values())
    counts = Counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_normalizer) as pool:
        for n, result in enumerate(pool.map(normalize_job, jobs),1):
            counts.update(result['counts'])
            if n % 10 == 0 or n == len(jobs):
                print(canonical({'jobs':n,'total_jobs':len(jobs),'counts':dict(counts)}),flush=True)
    save(STATE/('pilot-normalize-summary.json' if pilot else 'normalize-summary.json'),dict(counts))


def dedup():
    db = sqlite3.connect(STATE/'dedup.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY, record_id TEXT)')
    db.execute('DELETE FROM seen')
    counts = Counter()
    files = sorted((STATE/'normalized').glob('[0-9]*.jsonl'))
    writer, shard, shard_rows = None, -1, 0
    (STATE/'selected').mkdir(exist_ok=True)
    for path in files:
        if path.name.endswith('.excluded.jsonl'):
            continue
        for line in path.open():
            row = json.loads(line)
            try:
                db.execute('INSERT INTO seen VALUES (?,?)',(row['dedup_key'],row['record_id']))
            except sqlite3.IntegrityError:
                counts['exact_duplicate_episodes'] += 1
                continue
            if writer is None or shard_rows >= 2000:
                if writer:writer.close()
                shard += 1;shard_rows = 0
                writer = (STATE/'selected'/f'{shard:05d}.jsonl').open('w')
            writer.write(line);shard_rows+=1;counts['selected']+=1
        db.commit()
    if writer:writer.close()
    db.close()
    counts['shards'] = shard+1
    save(STATE/'dedup-summary.json',dict(counts))
    print(canonical(dict(counts)),flush=True)


def init_tokenizer():
    global PANG, MINI, RENDER
    from tokenizers import Tokenizer
    os.environ['TOKENIZERS_PARALLELISM']='false'
    PANG=Tokenizer.from_file(str(WORK/'data/blobs/tokenizer/tokenizer.json'))
    MINI=Tokenizer.from_file(str(RAW/'minicpm5-tokenizer/tokenizer.json'))
    RENDER=MiniCPMRenderer((RAW/'minicpm5-tokenizer/chat_template.jinja').read_text())


TOKEN_SCHEMA=pa.schema([('record_id',pa.string()),('input_ids',pa.list_(pa.int32())),
    ('labels',pa.list_(pa.int32())),('total_tokens',pa.int64()),('supervised_tokens',pa.int64())])


def compile_row(row):
    messages,tools=row['messages'],row['tools']
    ptext,pspans=pangolin_render(messages,tools)
    mtext,mspans=RENDER.render(messages,tools)
    pi,pl=tokenize_spans(PANG,ptext,pspans)
    mi,ml=tokenize_spans(MINI,mtext,mspans)
    # Reserve after the full trajectory: conservative for every earlier causal
    # assistant request too. Never truncate text or silently drop a tool turn.
    if max(len(pi),len(mi))+4096>131072:
        raise Rejected('context_budget_128k_with_4096_reserve')
    request=inference_messages(messages)
    mt,_=RENDER.render(request,tools,generation=True)
    if not mtext.startswith(mt):
        raise Rejected('native_answer_free_prefix')
    start=pspans[-1][0]
    pcompletion=ptext[:start]
    if row['record_id']!=sha(canonical({'version':'memory-harness-sft-v2','messages':messages,'tools':tools})):
        raise ValueError('record_identity_changed')
    record={k:row[k] for k in ('record_id','family_id','split','domain','source','upstream_source','license',
        'verification_tier','observation_origin','memory_placement','actual_retrieval',
        'recorded_error_followup','assistant_turns','tool_observations','query_hash')}
    record.update(messages_json=canonical(messages),tools_json=canonical(tools),memory_json=canonical(row['memory']),
        source_refs_json=canonical(row['source_refs']),skills_json=canonical(row['skills']),
        text=ptext,mini_text=mtext,target_spans_json=canonical(pspans),mini_target_spans_json=canonical(mspans),
        pangolin_tokens=len(pi),pangolin_supervised_tokens=sum(x!=-100 for x in pl),
        minicpm_tokens=len(mi),minicpm_supervised_tokens=sum(x!=-100 for x in ml),
        chat_prompt_sha256=sha(canonical(portable_messages(request,tools))),
        completion_prompt_sha256=sha(pcompletion),mini_prompt_sha256=sha(mt),
        output_reserve_tokens=4096,renderer_version='memory-harness-sft-v2-portable-json')
    token=lambda ids,labels:{'record_id':row['record_id'],'input_ids':ids,'labels':labels,
        'total_tokens':len(ids),'supervised_tokens':sum(x!=-100 for x in labels)}
    return record,token(pi,pl),token(mi,ml)


def compile_job(path):
    ident=Path(path).stem
    report=STATE/'compiled'/f'{ident}.json'
    if report.exists():return json.loads(report.read_text())
    report.parent.mkdir(exist_ok=True)
    counts=Counter();buffers={s:{'records':[],'tokenized':[],'minicpm5_tokenized':[]} for s in ('train','validation')}
    rejected=[]
    for line in Path(path).open():
        row=json.loads(line);counts['seen']+=1
        try:r,p,m=compile_row(row)
        except Rejected as exc:
            counts['excluded:'+str(exc)]+=1;rejected.append({'record_id':row['record_id'],'reason':str(exc)});continue
        dest=buffers[row['split']]
        for name,value in zip(('records','tokenized','minicpm5_tokenized'),(r,p,m)):dest[name].append(value)
        counts['accepted']+=1
        counts['pangolin_tokens']+=r['pangolin_tokens'];counts['minicpm_tokens']+=r['minicpm_tokens']
        counts['pangolin_supervised_tokens']+=r['pangolin_supervised_tokens'];counts['minicpm_supervised_tokens']+=r['minicpm_supervised_tokens']
    files=[]
    for split,configs in buffers.items():
        for name,rows in configs.items():
            if not rows:continue
            target=OUT/'data'/name/f'{split}-{ident}.parquet';target.parent.mkdir(parents=True,exist_ok=True)
            table=pa.Table.from_pylist(rows,schema=None if name=='records' else TOKEN_SCHEMA)
            pq.write_table(table,target,compression='zstd',compression_level=6,row_group_size=128)
            # Whole export readback, not just footer metadata.
            restored=pq.read_table(target)
            if not restored.equals(table):raise ValueError('parquet_readback_mismatch')
            files.append({'path':str(target.relative_to(OUT)),'bytes':target.stat().st_size,
                'sha256':file_sha(target),'rows':len(rows),'split':split,'config':name})
    result={'shard':ident,'counts':dict(counts),'files':files,'rejected':rejected,
            'all_exported_arrays_readback':True,'source_shard_sha256':file_sha(path)}
    save(report,result)
    return {k:v for k,v in result.items() if k!='rejected'}


def compile_all(workers,pilot=False):
    paths=sorted((STATE/('normalized' if pilot else 'selected')).glob('pilot-*.jsonl' if pilot else '*.jsonl'))
    paths=[p for p in paths if not p.name.endswith('.excluded.jsonl')]
    counts=Counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=init_tokenizer) as pool:
        for n,r in enumerate(pool.map(compile_job,paths),1):
            counts.update(r['counts'])
            print(canonical({'shards':n,'total':len(paths),'counts':dict(counts)}),flush=True)
    save(STATE/('pilot-compile-summary.json' if pilot else 'compile-summary.json'),dict(counts))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','normalize','dedup','compile'])
    p.add_argument('--workers',type=int,default=24);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if not 1<=a.workers<=32:raise ValueError('worker resource limit')
    if a.phase=='prepare':prepare()
    elif a.phase=='normalize':normalize(a.workers,a.pilot)
    elif a.phase=='dedup':dedup()
    else:compile_all(a.workers,a.pilot)


if __name__=='__main__':main()
