"""Also recognize source-faithful, exactly tied alternatives at a top-k cutoff."""
import json
from .full_source_visibility_v2 import check_source_visibility as check_previous,stable_scores
from .full_source_visibility import source_table,qa_task,prepare_tools,normalize_messages
from testtime.retrieval import Document


def tied_cutoff_observations(actual,expected,scores,documents):
    if not isinstance(actual,list) or not isinstance(expected,list) or not actual or len(actual)!=len(expected):return False
    if any(not isinstance(x,dict) or 'id' not in x for x in actual+expected):return False
    ids=[x['id'] for x in actual];expected_ids=[x['id'] for x in expected]
    if len(set(ids))!=len(ids) or len(set(expected_ids))!=len(expected_ids):return False
    if any(x!=documents.get(x['id']) for x in actual+expected):return False
    if any(i not in scores for i in ids+expected_ids):return False
    if any(scores[a]!=scores[b] for a,b in zip(ids,expected_ids)):return False
    changed=set(ids)^set(expected_ids)
    return all(scores[i]==scores[expected_ids[-1]] for i in changed)


def check_source_visibility(row):
    try:return check_previous(row)
    except ValueError as error:
        if str(error)!='QA visible question/documents/retrieval observations differ from pinned source reconstruction':raise
        original=error
    refs=json.loads(row['source_refs_json']);raws,_=source_table(row['source'],refs['file']);raw=raws[refs['row_index']]
    query,raw_messages,raw_tools,memory,_,_,_=qa_task(raw,row['source'])
    tools,validators=prepare_tools(raw_tools);expected=normalize_messages(raw_messages,tools,validators)
    actual=[m for m in json.loads(row['messages_json']) if m['role']!='system']
    if memory!=json.loads(row['memory_json']) or tools!=json.loads(row['tools_json']) or len(actual)!=len(expected):raise original
    docs=[Document(**d) for d in memory['documents']];scores=stable_scores(docs,query)
    originals={d.id:{'id':d.id,'title':d.title,'text':d.text,'source':d.source} for d in docs}
    changed=0;visible=set()
    for a,b in zip(actual[:-1],expected[:-1]):
        if a.get('role')=='tool':visible.update(x['id'] for x in json.loads(a['content']))
        if a==b:continue
        if a.get('role')!='tool' or b.get('role')!='tool':raise original
        if {k:v for k,v in a.items() if k!='content'}!={k:v for k,v in b.items() if k!='content'}:raise original
        if not tied_cutoff_observations(json.loads(a['content']),json.loads(b['content']),scores,originals):raise original
        changed+=1
    if not changed:raise original
    required=set(json.loads(expected[-1]['content'])['evidence_ids'])
    if not required<=visible:raise ValueError('Actual retrieval omits annotated source support')
    return 'exact_QA_source_evidence_with_fsum_equal_score_cutoff_ties'
