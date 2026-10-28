import hashlib
import json
import unittest
from barbet_inference.memory import canonical
from barbet_inference.value_admission import bind_contract_identity,disposition


class ValueAdmissionTests(unittest.TestCase):
    def test_prior_hold_wins_over_new_accept(self):
        self.assertEqual(disposition(query_held=True,sensitive=True,contextual_verdict='ACCEPT',bound_contract=True),'HOLD_PRIOR_QUERY')

    def test_negative_cannot_be_overridden_by_structural_proof(self):
        for verdict in ['REJECT','REVISE','HOLD']:
            self.assertEqual(disposition(query_held=False,sensitive=True,contextual_verdict=verdict,bound_contract=True),'HOLD_FIRST_NONACCEPT')

    def test_missing_review_is_not_approval(self):
        self.assertEqual(disposition(query_held=False,sensitive=True),'HOLD_PENDING_CONTEXTUAL_REVIEW')
        self.assertEqual(disposition(query_held=False,sensitive=False),'SCREEN_NEGATIVE_NOT_SEMANTICALLY_CERTIFIED')

    def test_system_and_complete_identity_required(self):
        system='Reviewed system';messages=[{'role':'system','content':system,'loss':False},{'role':'user','content':'question'},{'role':'assistant','content':'answer','loss':True}]
        trusted={hashlib.sha256(system.encode()).hexdigest()};identity=hashlib.sha256(canonical({'version':'memory-harness-sft-v2','messages':messages,'tools':[]}).encode()).hexdigest()
        row={'record_id':identity,'messages_json':json.dumps(messages),'tools_json':'[]'}
        self.assertIn(bind_contract_identity(row,trusted),trusted)
        messages[0]['content']='Unreviewed party obedience';row['messages_json']=json.dumps(messages)
        with self.assertRaises(ValueError):bind_contract_identity(row,trusted)
        messages[0]['content']=system;messages[-1]['content']='different answer';row['messages_json']=json.dumps(messages)
        with self.assertRaises(ValueError):bind_contract_identity(row,trusted)


if __name__=='__main__':unittest.main()
