import json
from pathlib import Path
import unittest
from barbet_inference.dataset_adapter import prepare_dataset_request
from barbet_inference.library import SkillLibrary
from barbet_inference.sft import pangolin_render, MiniCPMRenderer

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.messages=[{'role':'system','content':'rule','loss':False},
            {'role':'user','content':'first task','loss':False},
            {'role':'assistant','content':'FIRST_TARGET','loss':True},
            {'role':'user','content':'next task','loss':False},
            {'role':'assistant','content':'LAST_TARGET','loss':True}]
        text,spans=pangolin_render(self.messages)
        self.row={'messages_json':json.dumps(self.messages),'tools_json':'[]',
                  'text':text,'target_spans_json':json.dumps(spans)}

    def test_first_request_has_no_future_targets(self):
        r=prepare_dataset_request(self.row,model='m',target_turn=2)
        self.assertNotIn('TARGET',json.dumps(r))
        self.assertNotIn('next task',json.dumps(r))
        self.assertNotIn('loss',json.dumps(r))

    def test_last_request_keeps_real_prior_history(self):
        r=prepare_dataset_request(self.row,model='m')
        self.assertIn('FIRST_TARGET',json.dumps(r))
        self.assertNotIn('LAST_TARGET',json.dumps(r))

    def test_completion_is_exact_single_template_prefix(self):
        r=prepare_dataset_request(self.row,model='m',mode='pangolin_completion')
        self.assertTrue(r['prompt'].endswith('<|assistant_channel|>'))
        self.assertEqual(r['prompt'].count('<s>'),1)
        self.assertNotIn('LAST_TARGET',r['prompt'])
        self.assertFalse(r['add_special_tokens'])

    def test_nonassistant_target_rejected(self):
        with self.assertRaises(ValueError):prepare_dataset_request(self.row,model='m',target_turn=3)

    def test_skill_search_is_input_only_and_bounded(self):
        result=SkillLibrary().search('code python function tests debugging',top_k=2)
        self.assertEqual(len(result),2)
        self.assertTrue(all(r['selection_basis']=='query_only_lexical' for r in result))
        self.assertTrue(all(r['retrieval_score']>0 for r in result))

if __name__=='__main__':unittest.main()
