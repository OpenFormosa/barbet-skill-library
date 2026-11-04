import copy,json
from dataclasses import asdict
import pytest
from barbet_pipeline.common import canonical
from barbet_pipeline.full_graph_visibility import check_graph_visibility,question_for
from barbet_inference.memory import MemoryItem,compose_memory_prompt
from barbet_inference.relational_memory import Fact,RelationalMemory


def row_fixture(conceptnet=False,placement='front'):
    fact=Fact('f1','Q1','P1','Q2');state=RelationalMemory([fact])
    query={'operation':'project','branches':[{'start':'Q1','path':[{'relation':'P1','direction':'out'}]}]}
    doc=MemoryItem('f1',canonical({'subject':'Q1','relation':'P1','object':'Q2'}),'fixture:source')
    memory={'documents':[asdict(doc)],'registered_relation_snapshot':{
        'format':'barbet-finite-relations-v1','tool_name':'relation_lookup',
        'visibility':'registered_tool_state; observations become model-visible only after calls',
        'facts':[asdict(fact)],'snapshot_sha256':state.snapshot_sha256}}
    return {'source':('ConceptNet/' if conceptnet else 'Wikidata5M/')+'fixture',
        'source_refs_json':canonical({'query':query}),'memory_json':canonical(memory),
        'messages_json':canonical([{'role':'user','content':compose_memory_prompt(question_for(query,conceptnet),[doc],placement=placement)}]),
        'tools_json':'[]','memory_placement':placement}


@pytest.mark.parametrize('conceptnet,placement',[(False,'front'),(False,'back'),(True,'front'),(True,'back')])
def test_actual_visible_graph_matches_registered_state(conceptnet,placement):
    assert check_graph_visibility(row_fixture(conceptnet,placement))['operation']=='project'


def test_private_correct_oracle_does_not_hide_wrong_visible_edge():
    row=row_fixture();memory=json.loads(row['memory_json'])
    doc=memory['documents'][0];edge=json.loads(doc['text']);edge['object']='Q999';doc['text']=canonical(edge)
    query=json.loads(row['source_refs_json'])['query']
    row['memory_json']=canonical(memory)
    row['messages_json']=canonical([{'role':'user','content':compose_memory_prompt(question_for(query,False),[MemoryItem(**doc)],placement='front')}])
    with pytest.raises(ValueError,match='Visible graph edge'):check_graph_visibility(row)


def test_private_query_does_not_hide_wrong_visible_operation():
    row=row_fixture();messages=json.loads(row['messages_json'])
    messages[0]['content']=messages[0]['content'].replace('Return the distinct endpoint entity IDs when you','Count the distinct endpoint entity IDs when you')
    row['messages_json']=canonical(messages)
    with pytest.raises(ValueError,match='Visible graph operator'):check_graph_visibility(row)
