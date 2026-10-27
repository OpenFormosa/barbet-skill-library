import unittest
from barbet_inference.evidence_retrieval import retrieve_with_neighbors
from testtime.retrieval import Document


class EvidenceRetrievalTests(unittest.TestCase):
    def test_neighbor_keeps_dated_antecedent_and_negation(self):
        docs = [Document('a', 'The experiments began in 1938, not 1970.', 'source-one'),
                Document('b', 'Reward experiments trained animals to press a lever.', 'source-one'),
                Document('c', 'An unrelated source must not become a neighbor.', 'source-two')]
        selected, trace = retrieve_with_neighbors('When did reward experiments start?', docs,
            len, max_tokens=4000, top_k=1, neighbor_radius=1, target_chunk_chars=1, include_all_if_fits=False)
        text = '\n'.join(d.text for d in selected)
        self.assertIn('1938, not 1970', text)
        self.assertNotIn('An unrelated source', text)
        self.assertEqual([m['source_paragraph_ids'] for m in trace['selected']], [['a'], ['b']])

    def test_over_budget_paragraph_is_not_chopped(self):
        docs = [Document('a', 'Reward ' * 2000 + 'NOT a success.', 'source-one')]
        selected, trace = retrieve_with_neighbors('reward', docs, len, max_tokens=500, include_all_if_fits=False)
        self.assertEqual(selected, [])
        self.assertEqual(len(trace['whole_chunks_omitted_for_budget']), 1)
        self.assertFalse(trace['source_text_truncated'])

    def test_question_filler_does_not_displace_relevant_fact(self):
        docs = [Document('a', 'He was there and they were here, so how could they tell?', 'source-one'),
                Document('b', 'Monarch butterflies migrate south.', 'source-two')]
        selected, _ = retrieve_with_neighbors('Hey could you tell me how monarch butterflies migrate?', docs,
            len, top_k=1, neighbor_radius=0, include_all_if_fits=False)
        self.assertEqual(selected[0].source, 'source-two')

    def test_full_memory_fit_preserves_formula_and_actual_budget(self):
        docs = [Document('a', 'dy/dx = x/y. Do not swap the ratio.', 'source-one')]
        selected, trace = retrieve_with_neighbors('What is the ratio?', docs, len, max_tokens=2000)
        self.assertEqual(selected[0].text, docs[0].text)
        self.assertEqual(trace['method'], 'all_evidence_fits')
        self.assertLessEqual(trace['memory_tokens'], 2000)


if __name__ == '__main__': unittest.main()
