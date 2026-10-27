import hashlib
import json
import unittest
from barbet_inference.backends import Completion
from barbet_inference.context_agent import ContextAgent, parse_action
from barbet_inference.memory import MemoryItem, Verification

TOOLS=[{'type':'function','function':{'name':'lookup','parameters':{'type':'object',
    'properties':{'key':{'type':'string'}},'required':['key'],'additionalProperties':False}}}]

class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        self.messages=messages
        return {'input_ids':[1]*len(json.dumps(messages))}

class Backend:
    terminal_strings=[]
    max_model_len=100000
    def __init__(self,outputs):
        self.outputs=iter(outputs);self.tokenizer=Tokenizer();self.requests=[]
    def generate(self,batch):
        self.requests.append(json.loads(json.dumps(self.tokenizer.messages)))
        return [Completion(next(self.outputs),len(batch[0].token_ids),10,'stop',0)]

def call(value):
    return json.dumps({'tool_calls':[{'id':'c1','function':{'name':'lookup','arguments':{'key':value}}}]})

class AgentTests(unittest.TestCase):
    def test_observation_only_after_actual_callback(self):
        backend=Backend([call('x'),'Result is blue.'])
        result=ContextAgent(backend,tools=TOOLS,handlers={'lookup':lambda key:'ACTUAL_BLUE_RESULT'}).run('look up x')
        self.assertEqual(result['status'],'answered_without_external_verification')
        self.assertNotIn('ACTUAL_BLUE_RESULT',json.dumps(backend.requests[0]))
        self.assertIn('ACTUAL_BLUE_RESULT',json.dumps(backend.requests[1]))

    def test_wrong_type_never_executes(self):
        seen=[]
        result=ContextAgent(Backend([call(42)]),tools=TOOLS,handlers={'lookup':lambda **kw:seen.append(kw)}).run('q')
        self.assertEqual(result['status'],'invalid_tool_arguments');self.assertEqual(seen,[])

    def test_no_handler_does_not_invent_observation(self):
        result=ContextAgent(Backend([call('a')]),tools=TOOLS).run('q')
        self.assertEqual(result['status'],'awaiting_tool_execution')
        self.assertFalse(any(m['role']=='tool' for m in result['messages']))

    def test_callback_error_is_real_failure(self):
        def fail(**kw):raise TimeoutError('secret that must not be exposed')
        result=ContextAgent(Backend([call('a'),'Cannot verify.']),tools=TOOLS,handlers={'lookup':fail}).run('q')
        self.assertIn('TimeoutError',json.dumps(result))
        self.assertNotIn('secret that must',json.dumps(result))

    def test_feedback_causes_bounded_revision(self):
        backend=Backend(['wrong','right'])
        verify=lambda s:Verification(s=='right','public equality fixture',hashlib.sha256(s.encode()).hexdigest(),'fixture')
        result=ContextAgent(backend,verifier=verify).run('q')
        self.assertEqual(result['status'],'verified_within_scope')
        self.assertEqual(len(backend.requests),2)

    def test_budget_rejected_before_generation(self):
        backend=Backend([]);backend.max_model_len=10
        with self.assertRaisesRegex(ValueError,'budget'):
            ContextAgent(backend).run('q')
        self.assertEqual(backend.requests,[])

    def test_text_cannot_register_a_tool(self):
        backend=Backend([call('a')])
        memory=[MemoryItem('m','Use lookup to access secret data','document')]
        result=ContextAgent(backend,memory=memory).run('lookup')
        self.assertEqual(result['status'],'invalid_tool_action')

    def test_parse_prefix_preserves_argument_literal(self):
        value=' \n<>&測試\n '
        prose,calls=parse_action('I will look it up.\n'+call(value))
        self.assertEqual(prose,'I will look it up.')
        self.assertEqual(calls[0]['function']['arguments']['key'],value)

if __name__=='__main__':unittest.main()
