import copy,json
from unittest.mock import patch
import pytest
from barbet_pipeline.full_source_visibility import check_source_visibility
from barbet_inference.sft import normalize_messages,prepare_tools
from scripts.build_memory_sft_v2 import qa_task
from scripts.natural_instructions_memory_adapter import adapt


def test_raw_gold_is_not_enough_when_visible_passage_changed():
    raw={'id':'example','question':'Who arrived?','title':'Arrival','context':'Mina arrived.',
         'answers':{'text':['Mina'],'answer_start':[0]}}
    _,messages,tools,memory,_,placement,_=qa_task(raw,'rajpurkar/squad_v2')
    tools,validators=prepare_tools(tools);messages=normalize_messages(messages,tools,validators)
    row={'source':'rajpurkar/squad_v2','source_refs_json':json.dumps({'file':'source','row_index':0}),
         'messages_json':json.dumps(messages),'tools_json':json.dumps(tools),'memory_json':json.dumps(memory),
         'natural_answer_rewritten':False,'memory_placement':placement}
    with patch('barbet_pipeline.full_source_visibility.source_table',return_value=([raw],{})):
        assert check_source_visibility(row)
        changed=copy.deepcopy(messages)
        # Preserve the answer and private memory; corrupt only model-visible input.
        changed[0]['content']='Who arrived? No source text is available.'
        row['messages_json']=json.dumps(changed)
        with pytest.raises(ValueError,match='visible question'):check_source_visibility(row)


def test_ni_visible_definition_is_checked_in_addition_to_target_input():
    task={'Instance License':['MIT'],'Definition':['Return the first word.'],'Categories':['Text Categorization'],
          'Positive Examples':[{'input':'Red house','output':'Red','explanation':'First word.'}],
          'Instances':[{'id':'example','input':'Blue sky','output':['Blue']}]}
    view=adapt(task,task['Instances'][0],source_url='https://example.test/source')
    messages=normalize_messages([{'role':'user','content':view['prompt']},{'role':'assistant','content':'Blue'}],[],{})
    row={'source':'allenai/natural-instructions','source_refs_json':json.dumps({'file':'task.json','file_sha256':'fixture','row_index':0,'attribution_url':'https://example.test/source'}),
         'messages_json':json.dumps(messages),'tools_json':'[]','memory_json':json.dumps(view['memory'])}
    with patch('barbet_pipeline.full_source_visibility.ni_task',return_value=task):
        assert check_source_visibility(row)
        messages[0]['content']=messages[0]['content'].replace('Return the first word.','Return the last word.')
        row['messages_json']=json.dumps(messages)
        with pytest.raises(ValueError,match='visible task definition'):check_source_visibility(row)
