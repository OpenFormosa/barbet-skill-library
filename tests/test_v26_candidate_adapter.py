import copy
import unittest
from scripts.prepare_v26_answer_candidates import canonicalize,generic_styles,SCENARIO


class CandidateAdapterTests(unittest.TestCase):
    def row(self,system):
        return {'source':'Open-Orca/OpenOrca','family_id':'original-family','record_id':'original',
            'messages':[{'role':'system','content':system,'loss':False},{'role':'user','content':'Use this exact record: A=7. Return A.','loss':False},
                        {'role':'assistant','content':'7','loss':True}],
            'tools':[],'source_refs':{'transformations':[]}}

    def test_known_teacher_styles_share_condition_without_target_rewrite(self):
        styles=sorted(generic_styles());a=self.row('fixed skills'+SCENARIO+styles[0]);b=self.row('fixed skills'+SCENARIO+styles[1])
        ra=canonicalize(copy.deepcopy(a));rb=canonicalize(copy.deepcopy(b))
        self.assertEqual(ra['ppl_input_hash'],rb['ppl_input_hash'])
        self.assertEqual(ra['record_id'],rb['record_id'])
        self.assertEqual(ra['messages'][1:],a['messages'][1:])
        self.assertEqual(ra['source_family_id'],'original-family')

    def test_unknown_or_task_specific_system_constraints_are_preserved(self):
        a=self.row('fixed skills'+SCENARIO+'Output a JSON object with fields answer and evidence.')
        r=canonicalize(copy.deepcopy(a))
        self.assertEqual(r['messages'],a['messages'])
        self.assertNotIn('canonical_teacher_style_sha256',r['source_refs'])

    def test_other_sources_do_not_lose_identical_text(self):
        a=self.row('fixed skills'+SCENARIO+next(iter(generic_styles())));a['source']='different/repo'
        r=canonicalize(copy.deepcopy(a));self.assertEqual(r['messages'],a['messages'])


if __name__=='__main__':unittest.main()
