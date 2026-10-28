import unittest
from barbet_inference.refinement_evidence import observation_status, recorded_refinement_evidence


class RefinementEvidenceTests(unittest.TestCase):
    def test_error_word_or_null_field_is_not_an_error(self):
        for content in ['The docs explain error handling.', '{"error":null,"success":true}', '{"error":{"code":0,"message":""}}']:
            self.assertFalse(observation_status({'content':content,'name':'read_file'})[0])

    def test_same_tool_action_must_follow_an_actual_recorded_error(self):
        call=lambda ident:{'role':'assistant','loss':True,'content':'','tool_calls':[{'id':ident,'function':{'name':'lookup','arguments':{}}}]}
        obs=lambda ident,text:{'role':'tool','name':'lookup','tool_call_id':ident,'content':text}
        messages=[call('a'),obs('a','{"error":"missing value"}'),call('b'),obs('b','{"success":true}')]
        result=recorded_refinement_evidence(messages)
        self.assertEqual(result,{'explicit_error_observations':1,'same_tool_retries':1,
                                'retry_observations_without_explicit_error':1,'retry_observations_with_explicit_success':1})
        messages[1]['content']='{"error":null}'
        self.assertEqual(recorded_refinement_evidence(messages)['same_tool_retries'],0)

    def test_nonerror_payload_does_not_imply_explicit_success(self):
        self.assertEqual(observation_status({'content':'{"temperature":12}','name':'lookup'}),(False,False))


if __name__=='__main__':unittest.main()
