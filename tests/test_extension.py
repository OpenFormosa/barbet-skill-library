"""Authored fixtures, never extracted source prompts or reference answers."""
import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'routing'))
from extension import select

def inp(text, tools=None, previous=None):
    return {'messages':(previous or [])+[{'role':'user','content':text}], 'tools':tools or []}
def name(value):
    value=select(value)
    return value['skill'] if value else None
REGISTRY=[{'type':'function','function':{'name':'convert_units','description':'Convert physical units.',
    'parameters':{'type':'object','properties':{'value':{'type':'number'}},'required':['value']}}}]
CASES=[
 ('tool-capability-triage','Arrange a parcel delivery.', 'Thanks!',REGISTRY),
 ('external-state-grounding','Retrieve the stock quote for a named company.', 'Explain what a stock exchange does.',None),
 ('concept-explanation','Explain how a rainbow forms.', 'Write a poem about a rainbow.',None),
 ('comparison-by-dimensions','Compare induction and deduction.', 'Define induction.',None),
 ('constrained-creative-writing','Compose a four-line poem about fog.', 'What is the history of poetry?',None),
 ('text-transformation','Rewrite this: We opened the gate.', 'What is a passive verb?',None),
 ('translation-alignment','Translate "The gate is open" into French.', 'Explain the concept of translation.',None),
 ('practical-guidance','Suggest strategies to improve study habits.', 'How does the brain store memories?',None),
 ('option-generation','List four indoor activities.', 'Compute 7 * 8.',None),
 ('factual-question-resolution','Who invented the phonograph?', 'Generate a fictional inventor biography.',None),
 ('numeric-expression-evaluation','Evaluate (7 + 3) / 2.', 'Explain numerical stability.',None),
 ('descriptive-statistics','Find the median of [4, 1, 8].', 'What is a median?',None),
 ('ordered-top-k','Find the two smallest values in [6, 2, 8, 2].', 'Find the top 8 apps in a store.',None),
 ('classification-with-criteria','Classify this: delighted. Labels: positive, negative.', 'Explain text classification.',None),
 ('code-task-framing','Write a function to normalize a filename.', 'What is a function in mathematics?',None),
 ('hypothesis-and-evidence','Propose a hypothesis for a sudden change in pond color.', 'How do scientists test hypotheses?',None),
 ('constrained-sample-generation','Generate three 6-character sample strings.', 'Explain what a random variable is.',None),
 ('string-property-analysis','Check if "drawer" and "reward" are anagrams.', 'Compute cosine similarity of [2, 3] and [4, 6].',None),
 ('artifact-operation-contract','Resize an image to 320 by 240 pixels.', 'Describe how image compression works.',None),
 ('validation-scope-check','Validate the format of phone number +10000000000.', 'Write a function to validate a number.',None),
]
class Tests(unittest.TestCase):
    def test_every_authored_skill_has_activation_and_near_miss(self):
        for expected,positive,negative,tools in CASES:
            with self.subTest(skill=expected,kind='activation'):self.assertEqual(name(inp(positive,tools)),expected)
            with self.subTest(skill=expected,kind='near_miss'):self.assertNotEqual(name(inp(negative,tools)),expected)
        history=[{'role':'user','content':'Draft a short invitation.'},{'role':'assistant','content':'A first version.'}]
        self.assertEqual(name(inp('Another, please.',previous=history)),'conversation-reference-resolution')
        self.assertNotEqual(name(inp('Who invented the phonograph?',previous=history)),'conversation-reference-resolution')
    def test_forbidden_non_input_keys(self):
        for key in ['target','answer','future_messages','gold','source','split','score','metadata']:
            with self.subTest(key=key),self.assertRaises(ValueError):select(dict(inp('Explain gravity.'),**{key:'ignored?'}))
    def test_no_mutation_and_determinism(self):
        x=inp('Cancel an appointment.',REGISTRY);before=copy.deepcopy(x)
        self.assertEqual(select(x),select(copy.deepcopy(x)));self.assertEqual(x,before)
    def test_social_and_noise_have_no_fallback(self):
        for text in ['', 'Hi', 'No thanks.', 'Goodbye!', 'asdf_zzx', '...']:
            with self.subTest(text=text):self.assertIsNone(select(inp(text,REGISTRY if text!='asdf_zzx' and text!='...' else None)))
    def test_visible_result_is_not_new_request(self):
        self.assertIsNone(select({'messages':[{'role':'tool','content':'result'}],'tools':REGISTRY}))
    def test_existing_prefill_not_duplicated(self):
        prior=[{'role':'system','content':'[BARBET_SKILL_CONTEXT_V1] already supplied'}]
        self.assertIsNone(select(inp('Explain gravity.',previous=prior)))
    def test_ambiguous_capability_is_not_execution_authorization(self):
        s=select(inp('Do not call any tool. Explain why no tool fits this task.',REGISTRY))
        self.assertEqual(s['skill'],'tool-capability-triage');self.assertNotIn('tool_call',s)
    def test_missing_tool_description_never_invents_match(self):
        t=[{'function':{'name':'opaque','parameters':{}}}]
        self.assertEqual(name(inp('Arrange delivery.',t)),'tool-capability-triage')
    def test_order_of_operations_is_not_ordering_goods(self):
        self.assertNotEqual(name(inp('What math concepts use the order of operations?')),'external-state-grounding')
    def test_numbered_external_rank_is_not_supplied_list(self):
        self.assertEqual(name(inp("Find the top 9 paid apps in the 'Education' category.")),'external-state-grounding')
    def test_character_as_topic_is_not_random_string(self):
        self.assertNotEqual(name(inp('Generate a list of adjectives.\nCharacter')),'constrained-sample-generation')
    def test_topic_mention_is_not_creative_request(self):
        self.assertNotEqual(name(inp("That's an interesting headline! Can you give me five practical tips?")),'constrained-creative-writing')
    def test_prediction_mention_is_not_hypothesis_request(self):
        self.assertNotEqual(name(inp('How do scientists predict storms?')),'hypothesis-and-evidence')
        self.assertEqual(name(inp('Compare a hypothesis and a theory.')),'comparison-by-dimensions')
    def test_mechanism_is_not_advice(self):
        self.assertEqual(name(inp('How do sensors improve measurement accuracy?')),'concept-explanation')
    def test_new_question_does_not_inherit_topic(self):
        prior=[{'role':'user','content':'Write a tale.'},{'role':'assistant','content':'Which setting?'}]
        self.assertEqual(name(inp('That is fine. Who invented the phonograph?',previous=prior)),'factual-question-resolution')
    def test_skill_names_are_in_catalog(self):
        catalog={x['name'] for x in json.loads((ROOT/'catalog.json').read_text())['skills']}
        for expected,*_ in CASES:self.assertIn(expected,catalog)
        self.assertIn('conversation-reference-resolution',catalog)
    def test_no_external_access_or_eval(self):
        import ast
        tree=ast.parse((ROOT/'routing/extension.py').read_text())
        imports=[a.name for node in ast.walk(tree) if isinstance(node,ast.Import) for a in node.names]
        self.assertEqual(imports,['re'])
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'eval','exec','open','compile'} for n in ast.walk(tree)))
if __name__=='__main__':unittest.main()
