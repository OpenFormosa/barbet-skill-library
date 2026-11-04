import json,unittest
from barbet_pipeline.common import canonical,digest
from barbet_pipeline.full_requests import prepare_full_request
from barbet_inference.sft import portable_messages,pangolin_render

class FullRequestTest(unittest.TestCase):
    def row(self):
        messages=[{'role':'system','content':'Use the supplied context.','loss':False},
            {'role':'user','content':'What does the record say?','loss':False},
            {'role':'assistant','content':'ANSWER-NOT-IN-PROMPT','loss':True},
            {'role':'user','content':'FUTURE-NOT-IN-PROMPT','loss':False}]
        prefix=portable_messages(messages[:2],[])
        for message in prefix:message.pop('loss',None)
        prompt,_=pangolin_render(messages[:2],[]);prompt+='<|assistant_channel|>'
        return {'schema_version':'barbet-full-reprocess-v4','messages_json':canonical(messages),'tools_json':'[]',
            'private_oracle':'PRIVATE-NOT-IN-PROMPT','inference_decisions_json':canonical([{'target_turn':2,
                'chat_sha256':digest(prefix),'completion_sha256':digest(prompt),'output_reserve_tokens':4096}])}
    def test_only_public_causal_prefix_enters_both_modes(self):
        for mode in ['chat','completion']:
            request=prepare_full_request(self.row(),model='explicit-model',mode=mode)
            text=canonical(request)
            for forbidden in ['ANSWER-NOT-IN-PROMPT','FUTURE-NOT-IN-PROMPT','PRIVATE-NOT-IN-PROMPT']:self.assertNotIn(forbidden,text)
            self.assertEqual(request['max_tokens'],4096)
    def test_changed_input_or_unmeasured_turn_is_rejected(self):
        row=self.row();row['messages_json']=row['messages_json'].replace('record say','record contain')
        with self.assertRaises(ValueError):prepare_full_request(row,model='m')
        with self.assertRaises(ValueError):prepare_full_request(self.row(),model='m',target_turn=3)

if __name__=='__main__':unittest.main()
