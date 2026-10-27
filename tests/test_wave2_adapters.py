import copy
import json
import unittest
from barbet_inference.sft import Rejected, normalize_messages, prepare_tools
from scripts.memory_sft_wave2_adapters import toucan, openorca

TOOLS=[{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{'key':{'type':'string'}},'required':['key']}}}]


def sample():
    return {'uuid':'source','available_tools':TOOLS,'question_quality_assessment':{'question_quality':{'score':5}},
        'response_quality_assessment':{'completeness':{'score':5},'conciseness':{'score':3},'overall_score':4,'order_correctness':True,'desired_tools_used_percentage':1},
        'metadata':{'prompt_id':'1','mcp_servers':[{'server_id':42}]},
        'messages':[{'role':'user','content':'Look up the color for X.'},
                    {'role':'assistant','content':'','reasoning_content':'PRIVATE_REASONING'},
                    {'role':'assistant','content':'','function_call':{'name':'lookup','arguments':'{"key":"X"}'}},
                    {'role':'function','name':'lookup','content':'blue'},
                    {'role':'assistant','content':'It is blue.'}]}


class AdapterTests(unittest.TestCase):
    def test_real_function_observation_is_preserved_and_private_reasoning_removed(self):
        value=toucan(sample());tools,validators=prepare_tools(value['tools'])
        messages=normalize_messages(value['messages'],tools,validators)
        self.assertEqual(messages[2]['content'],'blue')
        self.assertEqual(messages[1]['tool_calls'][0]['id'],messages[2]['tool_call_id'])
        self.assertNotIn('PRIVATE_REASONING',json.dumps(value))

    def test_parallel_identical_names_are_never_guessed(self):
        row=sample();row['messages'].insert(3,copy.deepcopy(row['messages'][2]))
        with self.assertRaisesRegex(Rejected,'ambiguous'):toucan(row)

    def test_partial_upstream_task_is_held(self):
        row=sample();row['response_quality_assessment']['completeness']['score']=4
        with self.assertRaisesRegex(Rejected,'not_complete'):toucan(row)

    def test_exact_boilerplate_is_removed_but_scenario_rules_survive(self):
        row=sample();row['messages'].insert(0,{'role':'system','content':'<|im_system|>tool_declare<|im_middle|>'+json.dumps(TOOLS)+'<|im_end|>'})
        self.assertEqual(toucan(row)['messages'][0]['role'],'user')
        row['messages'][0]['content']='Do not alter the stored unit.'
        self.assertEqual(toucan(row)['messages'][0]['content'],'Do not alter the stored unit.')

    def test_trivia_is_not_claimed_as_retrieval_training(self):
        with self.assertRaisesRegex(Rejected,'no_supported_context'):
            openorca({'id':'flan.1','question':'What is the capital of France?','response':'Paris'})

    def test_provided_context_does_not_include_the_reference_answer(self):
        row={'id':'flan.1','question':'Read the following paragraph and identify the temperature: '+('The given operating procedure concerns device X. '*4)+'The limit is 35 degrees.','response':'TARGET_SENTINEL'}
        value=openorca(row)
        self.assertEqual(value['domain'],'reading')
        self.assertNotIn('TARGET_SENTINEL',json.dumps(value['messages'][:-1]))


if __name__=='__main__':unittest.main()
