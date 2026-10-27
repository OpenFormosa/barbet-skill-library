import math
import unittest
from types import SimpleNamespace

from barbet_inference.answer_selection import input_identity, select_minimum_ppl, NativeAnswerRenderer, score_batch


class AnswerSelectionTests(unittest.TestCase):
    def scored(self, rid, nll=2.0, **changes):
        return dict(record_id=rid, eligible=True, status='scored', input_hash='same',
                    protocol_sha256='pinned', mean_nll=nll, answer_tokens=3, **changes)

    def test_length_normalized_nll_wins_not_total_loss(self):
        a=self.scored('long',1.1);a.update(answer_tokens=100,nll_sum=110)
        b=self.scored('short',2.0);b.update(answer_tokens=2,nll_sum=4)
        self.assertEqual(select_minimum_ppl([a,b])['record_id'],'long')

    def test_failed_quality_cannot_win_with_lower_ppl(self):
        bad=self.scored('wrong',0.01);bad['eligible']=False
        self.assertEqual(select_minimum_ppl([bad,self.scored('valid')])['record_id'],'valid')

    def test_missing_or_nonfinite_competitor_holds_group(self):
        for patch in [{'status':'failed'},{'mean_nll':float('nan')},{'answer_tokens':0}]:
            missing=self.scored('missing');missing.update(patch)
            with self.assertRaisesRegex(ValueError,'missing or invalid'):
                select_minimum_ppl([self.scored('other'),missing])

    def test_different_condition_or_protocol_cannot_be_compared(self):
        for patch in [{'input_hash':'different'},{'protocol_sha256':'other'}]:
            c=self.scored('c');c.update(patch)
            with self.assertRaisesRegex(ValueError,'incomparable'):select_minimum_ppl([self.scored('a'),c])

    def test_stable_tie_and_answer_free_identity(self):
        self.assertEqual(select_minimum_ppl([self.scored('b'),self.scored('a')])['record_id'],'a')
        a=[{'role':'user','content':'task'},{'role':'assistant','content':'answer A'}]
        b=[a[0],{'role':'assistant','content':'answer B'}]
        self.assertEqual(input_identity(a),input_identity(b))
        self.assertNotEqual(input_identity(a, [{'name':'toolA'}]),input_identity(b,[{'name':'toolB'}]))
        self.assertNotEqual(input_identity(a),input_identity([{'role':'system','content':'different contract'},*b]))

    def test_unsupervised_assistant_history_is_part_of_condition(self):
        a=[{'role':'user','content':'task'},{'role':'assistant','content':'earlier fact A','loss':False}]
        b=[a[0],{'role':'assistant','content':'earlier fact B','loss':False}]
        self.assertNotEqual(input_identity(a),input_identity(b))

    def test_native_targets_exclude_prompt_and_include_eos(self):
        class CharTokenizer:
            def encode(self,text,add_special_tokens=False):
                return SimpleNamespace(ids=[ord(x) for x in text],offsets=[(i,i+1) for i in range(len(text))])
        template="{{bos_token}}{% for m in messages %}<{{m.role}}>{{m.content}}{% if m.role=='assistant' %}{{eos_token}}{% else %}{{'\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
        r=NativeAnswerRenderer(template,CharTokenizer()).render([{'role':'user','content':'long prompt'}, {'role':'assistant','content':'X','loss':True}])
        self.assertEqual(''.join(chr(x) for x in r['labels'] if x!=-100),'X</s>')

    def test_shift_and_padding_never_enter_answer_denominator(self):
        import torch
        class Backbone:
            embed_tokens=SimpleNamespace(weight=torch.zeros(1))
            def __call__(self,input_ids,**kwargs):
                return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids,7).float())
        class Model:
            model=Backbone()
            config=SimpleNamespace(pad_token_id=0)
            def lm_head(self,hidden):
                # Predict token 3 from predecessor 2; a one-token shift error fails.
                result=torch.zeros((len(hidden),7));result[:,3]=hidden[:,2]*2
                return result
        items=[dict(input_ids=[1,2,3],labels=[-100,-100,3],text_sha256='a',input_hash='a'),
               dict(input_ids=[1,1,1,2,3],labels=[-100,-100,-100,-100,3],text_sha256='b',input_hash='b')]
        r=score_batch(Model(),items)
        for x in r:
            self.assertEqual(x['answer_tokens'],1)
            self.assertAlmostEqual(x['mean_nll'],math.log(math.exp(2)+6)-2,places=6)


if __name__=='__main__':unittest.main()
