"""Source fidelity with a narrowly defined legacy floating-point tie allowance.

Legacy BM25 accumulates over a Python set. Different hash seeds can change the
last bit of tied scores and reorder equally ranked observations. Source text,
selected membership, tool arguments and all other visible content stay exact.
"""
from collections import defaultdict
import json,math
from .full_source_visibility import check_source_visibility as check_exact,source_table,qa_task,prepare_tools,normalize_messages
from testtime.retrieval import LexicalIndex,Document,terms


def stable_scores(documents,query):
    index=LexicalIndex(documents);contributions=defaultdict(list);n=len(documents)
    for word in sorted(set(terms(query))):
        posting=index.postings.get(word,())
        idf=math.log(1+(n-len(posting)+.5)/(len(posting)+.5))
        weight=.25 if len(word)==1 and '\u3400'<=word<='\u9fff' else 1.0
        if len(word)==2 and any(c in '的有在了與和及於幾' for c in word):weight*=.25
        for i,tf in posting:
            norm=1.2*(1-.75+.75*index.lengths[i]/index.average)
            contributions[i].append(weight*idf*tf*2.2/(tf+norm))
    return {documents[i].id:math.fsum(values) for i,values in contributions.items()}


def tied_observations(actual,expected,scores):
    if not isinstance(actual,list) or not isinstance(expected,list) or len(actual)!=len(expected):return False
    if any(not isinstance(x,dict) or 'id' not in x for x in actual+expected):return False
    a={x['id']:x for x in actual};b={x['id']:x for x in expected}
    if len(a)!=len(actual) or len(b)!=len(expected) or a!=b:return False
    # No threshold relaxation: stable fsum scores must be exactly equal at each
    # moved rank. A different selected set, quote, score or argument is rejected.
    return all(x['id']==y['id'] or (x['id'] in scores and y['id'] in scores and scores[x['id']]==scores[y['id']])
               for x,y in zip(actual,expected))


def check_source_visibility(row):
    try:return check_exact(row)
    except ValueError as error:
        if str(error)!='QA visible question/documents/retrieval observations differ from pinned source reconstruction':raise
        original=error
    refs=json.loads(row['source_refs_json']);raw_rows,_=source_table(row['source'],refs['file']);raw=raw_rows[refs['row_index']]
    query,raw_messages,raw_tools,memory,_,_,_=qa_task(raw,row['source'])
    tools,validators=prepare_tools(raw_tools);expected=normalize_messages(raw_messages,tools,validators)
    actual=[m for m in json.loads(row['messages_json']) if m['role']!='system']
    if memory!=json.loads(row['memory_json']) or tools!=json.loads(row['tools_json']) or len(actual)!=len(expected):raise original
    docs=[Document(**d) for d in memory['documents']];scores=stable_scores(docs,query);changed=0
    for a,b in zip(actual[:-1],expected[:-1]):
        if a==b:continue
        if a.get('role')!='tool' or b.get('role')!='tool':raise original
        if {k:v for k,v in a.items() if k!='content'}!={k:v for k,v in b.items() if k!='content'}:raise original
        if not tied_observations(json.loads(a['content']),json.loads(b['content']),scores):raise original
        changed+=1
    if not changed:raise original
    return 'exact_QA_source_content_and_selected_set_with_stable_fsum_equal_score_order'
