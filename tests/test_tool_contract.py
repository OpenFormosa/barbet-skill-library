import unittest
from jsonschema import Draft202012Validator
from barbet_inference.tool_contract import undeclared_arguments
from barbet_inference.context_agent import ContextAgent
from test_context_agent import Backend, call


class FunctionContractTests(unittest.TestCase):
    def test_schema_default_acceptance_is_not_a_declared_function_parameter(self):
        schema={'type':'object','properties':{'format':{'type':'string'}}}
        args={'format':'png','transparent':True}
        Draft202012Validator(schema).validate(args)
        self.assertEqual(undeclared_arguments(args,schema),['transparent'])
        self.assertNotIn('additionalProperties',schema)

    def test_explicit_open_objects_and_patterns_remain_valid(self):
        self.assertEqual(undeclared_arguments({'x':3},{'properties':{},'additionalProperties':True}),[])
        self.assertEqual(undeclared_arguments({'x-1':3},{'properties':{},'patternProperties':{'^x-':{'type':'integer'}}}),[])

    def test_composition_does_not_hide_declared_fields(self):
        schema={'properties':{'a':{}},'allOf':[{'properties':{'b':{}}}]}
        self.assertEqual(undeclared_arguments({'a':1,'b':2},schema),[])

    def test_unconfirmed_field_is_held_before_real_callback(self):
        seen=[]
        tools=[{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{}}}}]
        result=ContextAgent(Backend([call('undeclared')]),tools=tools,handlers={'lookup':lambda **kw:seen.append(kw)}).run('q')
        self.assertEqual(result['status'],'unconfirmed_tool_arguments');self.assertEqual(seen,[])


if __name__=='__main__':unittest.main()
