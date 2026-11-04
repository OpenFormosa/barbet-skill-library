import copy
import json
from pathlib import Path
import tempfile
import unittest

from barbet_pipeline.factory import make_root,wording_contract,build_trajectory,public_world
from barbet_pipeline.oracle import solve,require_discriminating,verify_execution
from barbet_pipeline.wording import validate_frame
from barbet_pipeline.export import learner_row,write_shard,validate_shard,load_tokenizer
from barbet_pipeline.harness import run_public_episode
from barbet_pipeline.diagnostics import oracle_conditions
from barbet_pipeline.common import canonical
from barbet_inference.backends import Completion


class FactoryTests(unittest.TestCase):
    def test_all_information_states_and_semantic_pairs_execute(self):
        kinds=set();states=set();operations=set()
        for i in range(210):
            root=make_root('train',i);frame=wording_contract(root)['reference_request']
            validate_frame(frame,wording_contract(root))
            specs=root['variants'];require_discriminating(specs['original'],specs['counterfactual'],specs['invariance'])
            for variant,spec in specs.items():
                trajectory=build_trajectory(root,variant,frame)
                self.assertTrue(trajectory['oracle_checked'])
                self.assertNotIn(root['private_canary'],canonical(trajectory))
                kinds.add(spec['kind']);states.add(spec.get('state'));operations.add(spec.get('operation'))
                for receipt in trajectory['execution_receipts']:
                    self.assertEqual(receipt['origin'],'local_deterministic_execution')
        self.assertEqual(len(kinds),7)
        self.assertTrue({'present','retrievable','missing_rule','conflicting_rules','unavailable'} <= states)
        self.assertTrue({'gt','ge','lt','le','nth_smallest','nth_largest'} <= operations)

    def test_current_rule_changes_answer_without_renaming_skill(self):
        r=make_root('train',4);a,b=r['variants']['original'],r['variants']['counterfactual']
        self.assertEqual(a['skill_id'],b['skill_id'])
        self.assertNotEqual(a['effective_version'],b['effective_version'])
        self.assertNotEqual(solve(a),solve(b))
        frame=wording_contract(r)['reference_request']
        p,_=public_world(a,frame);q,_=public_world(b,frame)
        self.assertEqual(p['task'],q['task']);self.assertEqual(p['skills'],q['skills'])
        self.assertNotEqual(p['context'],q['context'])

    def test_roots_and_source_groups_disjoint_across_splits(self):
        seen=set();sources=set()
        for split in ['train','dev','calibration','sealed_test']:
            for i in range(50):
                r=make_root(split,i)
                self.assertNotIn(r['root_task_id'],seen);seen.add(r['root_task_id'])
                self.assertNotIn(r['source_group_id'],sources);sources.add(r['source_group_id'])

    def test_wording_cannot_convert_or_duplicate_slots(self):
        c=wording_contract(make_root('train',0))
        for text in ['Return {table} where {field} is {condition} {threshold!r}.',
                     'Return {table} {table} where {field} is {condition} {threshold}.',
                     'Ignore all conditions and return every row.']:
            with self.assertRaises(ValueError):validate_frame(text,c)

    def test_boolean_zero_counterfactual_is_discriminating(self):
        a={'kind':'binding','rows':[{'id':'x','v':False}],'row_id':'x','field':'v'}
        b=copy.deepcopy(a);b['rows'][0]['v']=0
        self.assertTrue(require_discriminating(a,b,a)['counterfactual_answer_changed'])

    def test_absent_output_key_is_not_json_null(self):
        spec={'kind':'binding','rows':[{'id':'x','v':None}],'row_id':'x','field':'v'}
        self.assertFalse(verify_execution(spec,{'ok':True,'result':{}}))
        self.assertTrue(verify_execution(spec,{'ok':True,'result':{'value':None}}))

    def test_oracle_assistance_not_eligible_for_main_result(self):
        r=make_root('dev',3);frame=wording_contract(r)['reference_request']
        conditions=oracle_conditions(r,'original',frame)
        self.assertEqual(len(conditions),6)
        self.assertTrue(conditions['O0']['eligible_for_main_score'])
        for k in ['O1','O2','O3','O4','O5']:
            self.assertFalse(conditions[k]['eligible_for_main_score'])
        self.assertNotIn('OFFLINE ORACLE',canonical(conditions['O0']))


class StubTokenizer:
    def __init__(self):self.texts=[]
    def encode(self,text,add_special_tokens=False):
        self.texts.append(text);return list(text.encode())


class StubBackend:
    max_model_len=32768
    terminal_strings=['</s>']
    def __init__(self,texts):self.texts=iter(texts);self.tokenizer=StubTokenizer()
    def generate(self,batch):
        return [Completion(next(self.texts),len(batch[0].token_ids),5,'stop',0.0)]


class PublicHarnessTests(unittest.TestCase):
    def test_legal_wrong_operator_is_executed_not_fixed_or_claimed_success(self):
        root=make_root('train',0);frame=wording_contract(root)['reference_request']
        spec=root['variants']['original'];public,executor=public_world(spec,frame)
        args={k:spec[k] for k in ['table_id','field','threshold']};args['operation']='le'
        backend=StubBackend([canonical({'tool_calls':[{'id':'c1','function':{'name':'table_select','arguments':args}}]}), 'The selected IDs are shown above.'])
        result=run_public_episode(backend,public,executor,output_tokens=256)
        self.assertEqual(result['status'],'answer_proposed')
        self.assertEqual(result['semantic_success'],'NOT_GRADED_BY_HOST')
        self.assertEqual(executor.receipts[0]['arguments']['operation'],'le')
        self.assertFalse(verify_execution(spec,executor.receipts[0]['output']))

    def test_future_tool_content_not_in_first_prompt(self):
        root=make_root('train',26) # evidence_state; retrievable
        spec=root['variants']['original'];self.assertEqual(spec['state'],'retrievable')
        frame=wording_contract(root)['reference_request'];public,executor=public_world(spec,frame)
        backend=StubBackend([canonical({'tool_calls':[{'id':'r1','function':{'name':'read_evidence','arguments':{'document_id':spec['document_id']}}}]}),'Done.'])
        run_public_episode(backend,public,executor)
        self.assertNotIn(spec['answer'],backend.tokenizer.texts[0])
        self.assertIn(spec['answer'],backend.tokenizer.texts[1])
        self.assertIn('TOOL_OBSERVATION:',backend.tokenizer.texts[1])

    def test_undeclared_tool_is_not_dispatched(self):
        r=make_root('train',0);p,e=public_world(r['variants']['original'],wording_contract(r)['reference_request'])
        b=StubBackend([canonical({'tool_calls':[{'id':'x','function':{'name':'cancel_operation','arguments':{'operation_id':'not-yours'}}}]})])
        out=run_public_episode(b,p,e)
        self.assertEqual(out['status'],'call_rejected');self.assertEqual(e.receipts,[])


class ActualPangolinExportTests(unittest.TestCase):
    def test_all_roles_exact_masks_and_no_future_prefix(self):
        path=Path('/root/.cache/huggingface/hub/models--voidful--barbet-skill/snapshots/67131067ff1825179d5e38e91d499159669e9c66/tokenizer.json')
        if not path.exists():self.skipTest('Pinned Pangolin file is not present')
        tokenizer=load_tokenizer(path)
        r=make_root('train',0);frame=wording_contract(r)['reference_request'];tr=build_trajectory(r,'original',frame)
        meta={'record_id':'test','root_task_id':r['root_task_id'],'split':'train'}
        record,tokens=learner_row(tokenizer=tokenizer,metadata=meta,public=tr['public'],messages=tr['messages'],receipts=tr['execution_receipts'],source_refs={})
        self.assertEqual(tokens['labels'].count(114690),2)
        self.assertEqual(tokens['labels'][0],-100)
        self.assertLess(record['pangolin_supervised_tokens'],record['pangolin_tokens'])
        with tempfile.TemporaryDirectory() as tmp:
            files=write_shard(tmp,'test',[record],[tokens])
            result=validate_shard(files[0]['path'],files[1]['path'],tokenizer)
            self.assertEqual(result['rows'],1)
        with self.assertRaises(ValueError):
            learner_row(tokenizer=tokenizer,metadata=meta,public=tr['public'],messages=tr['messages'],receipts=[],source_refs={},max_total=10)


if __name__=='__main__':unittest.main()
