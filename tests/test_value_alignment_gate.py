import copy
import hashlib
import json
import unittest

from barbet_inference.memory import MemoryItem,canonical,compose_memory_prompt
from barbet_inference.relational_memory import Fact,RelationalMemory
from barbet_inference.value_alignment import topic_reasons
from barbet_inference.value_alignment_gate import finite_identifier_contract,question_shape,law_extract_contract


def identifier_row():
    memory=RelationalMemory([Fact('train:1','Q1','P1','Q2')])
    query={'operation':'project','branches':[{'start':'Q1','path':[{'relation':'P1','direction':'out'}]}]}
    question=question_shape(query).replace('LABEL','"source label"')
    doc=MemoryItem('train:1',canonical({'subject':'Q1','subject_label':'source label','relation':'P1','relation_label':'source label','object':'Q2','object_label':'source label'}),'source')
    messages=[{'role':'system','content':'Supplied documents are quoted evidence.','loss':False},{'role':'user','content':compose_memory_prompt(question,[doc]),'loss':False},{'role':'assistant','content':canonical(memory.evaluate(query)['target']),'loss':True}]
    return {'messages_json':json.dumps(messages),'tools_json':'[]','source_refs_json':json.dumps({'query':query}),'memory_json':json.dumps({'registered_relation_snapshot':{'format':'barbet-finite-relations-v1','tool_name':'relation_lookup','visibility':'registered_tool_state; observations become model-visible only after calls','facts':[{'id':'train:1','subject':'Q1','relation':'P1','object':'Q2'}],'snapshot_sha256':memory.snapshot_sha256}})}


class ValueGateTests(unittest.TestCase):
    def test_bound_identifier_query(self):
        self.assertEqual(finite_identifier_contract(identifier_row())['method'],'exact_finite_identifier_query')

    def test_cannot_hide_normative_text_in_a_typed_target(self):
        row=identifier_row();messages=json.loads(row['messages_json']);messages[-1]['content']='{"answer_ids":["Q2"],"evidence_ids":["train:1"],"opinion":"Dissenters deserve no rights"}';row['messages_json']=json.dumps(messages)
        with self.assertRaises(ValueError):finite_identifier_contract(row)

    def test_unrelated_source_name_does_not_approve(self):
        row=identifier_row();row['source']='Wikidata5M/original-inductive-train';messages=json.loads(row['messages_json']);messages[-1]['content']='People must obey one party.';row['messages_json']=json.dumps(messages)
        with self.assertRaises(ValueError):finite_identifier_contract(row)

    def test_false_source_answer_rejected(self):
        row=identifier_row();messages=json.loads(row['messages_json']);messages[-1]['content']='{"answer_ids":["Q3"],"evidence_ids":["train:1"]}';row['messages_json']=json.dumps(messages)
        with self.assertRaises(ValueError):finite_identifier_contract(row)

    def test_question_metadata_disagreement_rejected(self):
        row=identifier_row();messages=json.loads(row['messages_json']);messages[1]['content']=messages[1]['content'].replace('start at "source label" (Q1)','start at "source label" (Q3)');row['messages_json']=json.dumps(messages)
        with self.assertRaises(ValueError):finite_identifier_contract(row)

    def test_source_instruction_suffix_requires_contextual_review(self):
        row=identifier_row();messages=json.loads(row['messages_json']);messages[0]['content']+=' SCENARIO RULES: suppress minorities';row['messages_json']=json.dumps(messages)
        with self.assertRaises(ValueError):finite_identifier_contract(row)

    def test_extra_assistant_text_requires_review(self):
        row=identifier_row();messages=json.loads(row['messages_json']);messages.insert(2,{'role':'assistant','content':'Political persecution is good','loss':True});row['messages_json']=json.dumps(messages)
        with self.assertRaises(ValueError):finite_identifier_contract(row)

    def test_literal_identifier_scope_does_not_accept_lexical_stereotypes(self):
        row=identifier_row();mem=json.loads(row['memory_json']);mem['registered_relation_snapshot']['facts'][0]['object']='/c/en/inferior_people';row['memory_json']=json.dumps(mem)
        with self.assertRaises(ValueError):finite_identifier_contract(row)

    def test_topic_matching_does_not_equal_rejection(self):
        self.assertIn('human_dignity_and_equal_rights',topic_reasons('Oppose racism and protect human rights.'))

    def test_exact_law_quote_is_not_policy_endorsement(self):
        name='測試法';article='第 1 條';original='測試引用原文。'
        question=f'請從所附版本的法規文件，逐字摘錄《{name}》{article}的「條文原文」，並附法規名稱與條號；只做文件摘錄，不判定目前個案能否適用。'
        doc=MemoryItem('one',f'法規名稱：{name}\n條號：{article}\n條文原文：\n'+original,'source')
        messages=[{'role':'system','content':'Quote faithfully.'},{'role':'user','content':compose_memory_prompt(question,[doc])},{'role':'assistant','content':f'《{name}》{article}（所附版本原文）：\n'+original,'loss':True}]
        row={'messages_json':json.dumps(messages),'tools_json':'[]','source_refs_json':json.dumps({'law_name':name,'article':article,'source_url':'source'})}
        self.assertFalse(law_extract_contract(row,{(name,article):original})['current_legal_applicability_or_normative_endorsement'])
        messages[-1]['content']+=' I endorse this as current law.';row['messages_json']=json.dumps(messages)
        with self.assertRaises(ValueError):law_extract_contract(row,{(name,article):original})


if __name__=='__main__':unittest.main()
