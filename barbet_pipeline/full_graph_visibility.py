"""Bind finite-graph oracle operations and state to the actual visible task."""
import json,re
from .common import canonical
from barbet_inference.memory import MemoryItem,memory_block
from barbet_inference.relational_dataset import restore_relational_memory


def question_for(query,conceptnet):
    op=query['operation'];key='count' if op=='count' else 'answer_ids'
    if conceptnet:
        return ('Query these finite ConceptNet 5.7 lexical records using the exact URI strings in this specification: '+canonical(query)+'. '
            'Each branch starts from its start URI and follows its path in order. direction out means subject to object. '
            'Keep sense and part-of-speech suffixes; do not replace a relation, invert an edge or infer additional facts. '
            'project returns distinct endpoints of the first branch; count counts those endpoints; union, intersection and difference '
            'mean A union B, A intersection B and A minus B for the first and second branches. '
            f'Return one JSON object with {key} and evidence_ids. '+
            ('count is an integer. ' if op=='count' else 'answer_ids is a lexicographically sorted array of complete URI strings. ')+
            'evidence_ids is the sorted union of every matching record ID traversed at every step of every branch, '
            'including traversed records whose next step has no match. Exclude unrelated records. '
            'An absent match means absent from this provided snapshot, not false in the world. '
            'This is a query over attributed source assertions, not a request to assert that every lexical relation is universally true.')
    descriptions=[]
    for branch in query['branches']:
        text=f'start at "<label>" ({branch["start"]})'
        for step in branch['path']:
            if step['direction']!='out':raise ValueError('Wikidata template only defines outgoing relations')
            text+=f', then follow the outgoing "<label>" ({step["relation"]}) relation'
        descriptions.append(text)
    if op=='project':task='Return the distinct endpoint entity IDs when you '+descriptions[0]+'.'
    elif op=='count':task='Count the distinct endpoint entity IDs when you '+descriptions[0]+'.'
    else:
        task='Build set A when you '+descriptions[0]+'. Build set B when you '+descriptions[1]+'. '
        task+={'union':'Return all IDs in A or B, with duplicates removed.',
               'intersection':'Return only IDs present in both A and B.',
               'difference':'Return IDs in A that are absent from B.'}[op]
    return ('Use only the supplied finite relation records, not outside knowledge or current-world assumptions. '
            'Follow directed relations exactly; repeated endpoints count once. '+task+
            f' Return one JSON object with {key} and evidence_ids. '+
            ('count is an integer. ' if op=='count' else 'answer_ids is a lexicographically sorted array of entity IDs. ')+
            'evidence_ids must be the sorted union of all matching record IDs traversed at every step of every branch, '
            'including traversed records whose branches end without a further match. Do not include unrelated records. '
            'An empty result means no match in these supplied records; it does not mean false in the world.')


def check_graph_visibility(row):
    source=row['source']
    if not source.startswith(('Wikidata5M/','ConceptNet/')):return None
    conceptnet=source.startswith('ConceptNet/')
    query=json.loads(row['source_refs_json'])['query'];metadata=json.loads(row['memory_json'])
    messages=json.loads(row['messages_json']);tools=json.loads(row['tools_json'])
    users=[m['content'] for m in messages if m['role']=='user']
    if len(users)!=1:raise ValueError('Unexpected graph task continuation')
    text=users[0];state=restore_relational_memory(row);documents=metadata['documents']
    if tools:
        if documents or row['memory_placement']!='chronological':raise ValueError('Graph tool/inline memory contract changed')
        suffix=' Read the registered records with relation_lookup before answering.' if conceptnet else ' The records are available through relation_lookup; read them before answering.'
        if not text.endswith(suffix):raise ValueError('Graph tool instruction differs')
        question=text[:-len(suffix)]
    else:
        items=[MemoryItem(**d) for d in documents];block=memory_block(items,turn=0)
        if row['memory_placement']=='front':
            prefix=block+'\n\nQUESTION:\n'
            if not text.startswith(prefix):raise ValueError('Actual graph evidence differs from public memory')
            question=text[len(prefix):]
        elif row['memory_placement']=='back':
            suffix='\n\n'+block
            if not text.startswith('QUESTION:\n') or not text.endswith(suffix):raise ValueError('Actual graph evidence differs from public memory')
            question=text[len('QUESTION:\n'):-len(suffix)]
        else:raise ValueError('Unsupported inline graph placement')
        actual={d.id:json.loads(d.text) for d in items}
        expected={f.id:(f.subject,f.relation,f.object) for f in state.facts}
        if len(actual)!=len(items) or set(actual)!=set(expected):raise ValueError('Visible facts differ from registered graph state')
        if any(tuple(actual[i].get(k) for k in ['subject','relation','object'])!=v for i,v in expected.items()):
            raise ValueError('Visible graph edge differs from independently evaluated state')
    if not conceptnet:
        # Explicit Q/P IDs determine these finite tasks. Display labels are not
        # claimed as independently verified current-world entity facts.
        question=re.sub(r'"(?:\\.|[^"\\])*"','"<label>"',question)
    if question!=question_for(query,conceptnet):raise ValueError('Visible graph operator, direction or entity binding differs from evaluated query')
    return {'source':source,'operation':query['operation'],'path_lengths':','.join(str(len(b['path'])) for b in query['branches']),
            'memory_placement':row['memory_placement'],'visible_inline_facts':len(documents)}
