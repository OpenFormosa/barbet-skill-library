import copy
import unittest
from scripts.guard_v26_episode_identity import identities
from scripts.prepare_v26_answer_candidates import canonicalize
from barbet_inference.answer_selection import input_identity


class EpisodeIdentityTests(unittest.TestCase):
    def row(self,cid='source_id'):
        return {'source':'fixture/repo','family_id':'family','source_refs':{'transformations':[]},
            'tools':[{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{}}}}],
            'messages':[{'role':'user','content':'Look up the blue box.','loss':False},
                {'role':'assistant','content':'','loss':True,'tool_calls':[{'id':cid,'type':'function','function':{'name':'lookup','arguments':{}}}]},
                {'role':'tool','name':'lookup','tool_call_id':cid,'content':'17 bolts','loss':False},
                {'role':'assistant','content':'The blue box has 17 bolts.','loss':True}]}

    def test_result_variants_are_same_task_but_different_conditions(self):
        a=self.row();b=copy.deepcopy(a);b['messages'][2]['content']='19 bolts'
        self.assertEqual(identities(a)[0],identities(b)[0])
        self.assertNotEqual(input_identity(a['messages'],a['tools']),input_identity(b['messages'],b['tools']))

    def test_question_or_capability_changes_do_not_merge_tasks(self):
        a=self.row();b=copy.deepcopy(a);b['messages'][0]['content']='Look up the red box.'
        self.assertNotEqual(identities(a)[0],identities(b)[0])
        c=copy.deepcopy(a);c['tools'][0]['function']['description']='Different documented environment'
        self.assertNotEqual(identities(a)[0],identities(c)[0])

    def test_exact_prefix_match_survives_provider_call_id_aliases(self):
        full=self.row('full-provider-id');short=self.row('short-provider-id');short['messages']=short['messages'][:2]
        short=canonicalize(short)
        self.assertIn(short['candidate_identity'],identities(full)[1])
        different=copy.deepcopy(short);different['messages'][1]['tool_calls'][0]['function']['arguments']={'x':2}
        self.assertNotIn(canonicalize(different)['candidate_identity'],identities(full)[1])


if __name__=='__main__':unittest.main()
