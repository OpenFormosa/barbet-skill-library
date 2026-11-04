import copy,json,unittest
from barbet_pipeline.full_quality import independent_graph_target,QualityHold
from barbet_inference.relational_memory import Fact
from barbet_pipeline.full_corpus import refresh_singleton_skills

class FullCorpusQualityTest(unittest.TestCase):
    def setUp(self):
        self.facts=[Fact('1','a','p','b'),Fact('2','a','p','c'),Fact('3','b','q','d'),Fact('4','x','p','d')]
        self.branch={'start':'a','path':[{'relation':'p','direction':'out'},{'relation':'q','direction':'out'}]}
    def test_traversed_dead_end_evidence_remains(self):
        self.assertEqual(independent_graph_target(self.facts,{'operation':'project','branches':[self.branch]}),{'answer_ids':['d'],'evidence_ids':['1','2','3']})
    def test_count_is_not_number_of_traversed_edges(self):
        self.assertEqual(independent_graph_target(self.facts,{'operation':'count','branches':[self.branch]})['count'],1)
    def test_difference_is_directional(self):
        second={'start':'x','path':[{'relation':'p','direction':'out'}]}
        result=independent_graph_target(self.facts,{'operation':'difference','branches':[self.branch,second]})
        self.assertEqual(result,{'answer_ids':[],'evidence_ids':['1','2','3','4']})
    def test_inverse_edge_uses_correct_endpoint(self):
        b={'start':'d','path':[{'relation':'q','direction':'in'}]}
        self.assertEqual(independent_graph_target(self.facts,{'operation':'project','branches':[b]})['answer_ids'],['b'])
    def test_wrong_arity_is_not_silently_ignored(self):
        with self.assertRaises(QualityHold):independent_graph_target(self.facts,{'operation':'count','branches':[self.branch,self.branch]})
    def test_unknown_operator_does_not_fall_back(self):
        with self.assertRaises(QualityHold):independent_graph_target(self.facts,{'operation':'xor','branches':[self.branch,self.branch]})
    def test_ranked_input_is_preserved_exactly(self):
        row={'source_refs_json':json.dumps({'answer_selection':{'competitors':2}})}
        messages=[{'role':'system','content':'frozen ranked input'}]
        result,metadata=refresh_singleton_skills(row,messages)
        self.assertEqual(result,messages);self.assertIsNone(metadata)
    def test_singleton_refresh_keeps_values_and_question(self):
        row={'source_refs_json':json.dumps({'answer_selection':{'competitors':1}}),'skills_json':'["grounded-evidence-answer"]'}
        messages=[{'role':'system','content':'Header\n\nAPPLICABLE SKILLS:\nold\n明示價值設定：台灣人民自決。'}, {'role':'user','content':'original question'}]
        result,metadata=refresh_singleton_skills(row,messages)
        self.assertIn('台灣人民自決',result[0]['content']);self.assertEqual(result[1],messages[1])
        self.assertEqual(metadata[0]['name'],'grounded-evidence-answer')
        self.assertIn('old',messages[0]['content'])

if __name__=='__main__':unittest.main()
