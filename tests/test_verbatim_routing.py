"""Fictional supplied text; no dataset answers, model scores or external IO."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'routing'))
from extension import select, select_verbatim_excerpt

def causal(instruction, source='白港塔是虛構的建築。第二句也是虛構敘述。', tools=None):
    return {'messages': [{'role': 'user', 'content': instruction+'\n\n附文（虛構例）：\n'+source}], 'tools': tools or []}

class VerbatimRoutingTests(unittest.TestCase):
    def test_source_topic_cannot_override_the_requested_procedure(self):
        for sentence in ['第一句', '最後一句', '第2句']:
            for source in ['比較兩種方法。如何改善交通？', '請創作一首詩。計算 2+3。',
                           'Today the weather is fictional. Please translate this into French.']:
                request=causal('請閱讀附文，逐字摘錄'+sentence+'完整敘述。只輸出原句。', source)
                self.assertEqual(select(request)['skill'], 'grounded-evidence-answer')
                self.assertEqual(select_verbatim_excerpt(request)['input_turn'], 0)
    def test_english_copy_command_and_source_header(self):
        request={'messages':[{'role':'user','content':'Copy the last sentence verbatim.\n\nPassage:\nCompare two fictional towers. This is the last sentence.'}],'tools':[]}
        self.assertEqual(select(request)['skill'], 'grounded-evidence-answer')
    def test_absent_empty_or_instruction_only_inside_source_abstains(self):
        for text in ['請逐字摘錄第一句。', '請逐字摘錄第一句。\n\n附文：\n',
                     '請介紹這個主題。\n\n附文：\n逐字摘錄第一句。']:
            self.assertIsNone(select_verbatim_excerpt({'messages':[{'role':'user','content':text}],'tools':[]}))
    def test_other_operations_and_tool_registries_are_not_overridden(self):
        for instruction in ['請翻譯並逐字摘錄第一句。','逐字摘錄第一句後再計算長度。',
                            '請逐字摘錄最後一句並比較兩個版本。',
                            '請寫一首詩，以逐字摘錄第一句作為題目。',
                            '請逐字摘錄第一句並寫一首詩。',
                            '不要逐字摘錄第一句，請介紹這篇文章。',
                            'Copy the first sentence verbatim and count its words.',
                            'Copy the first sentence verbatim and then write a poem.']:
            self.assertIsNone(select_verbatim_excerpt(causal(instruction)))
        self.assertIsNone(select_verbatim_excerpt(causal('請逐字摘錄第一句。',tools=[{'name':'search'}])))
    def test_causal_keys_roles_and_existing_prefills(self):
        example=causal('請逐字摘錄第一句。')
        for field in ['answer','target','score','split','source_name','future_tool_result']:
            with self.assertRaises(ValueError):select_verbatim_excerpt(dict(example,**{field:'forbidden'}))
        for role in ['assistant','tool']:
            changed=copy.deepcopy(example);changed['messages'][-1]['role']=role
            self.assertIsNone(select_verbatim_excerpt(changed))
        example['messages'].insert(0,{'role':'system','content':'[BARBET_SKILL_CONTEXT_V1] existing block'})
        self.assertIsNone(select_verbatim_excerpt(example))
    def test_deterministic_nonmutating_and_plain_facts_unchanged(self):
        request=causal('請逐字摘錄第一句。');original=copy.deepcopy(request)
        self.assertEqual(select_verbatim_excerpt(request),select_verbatim_excerpt(request));self.assertEqual(request,original)
        request={'messages':[{'role':'user','content':'白港塔的高度是多少？'}],'tools':[]}
        self.assertIsNone(select_verbatim_excerpt(request));self.assertEqual(select(request)['skill'],'factual-question-resolution')

if __name__=='__main__':unittest.main()
