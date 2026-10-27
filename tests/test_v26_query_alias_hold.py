import unittest
from scripts.prune_v26_query_aliases_v1 import keep_indices


class QueryAliasHoldTests(unittest.TestCase):
    def test_all_answer_views_are_held_without_lowest_unequal_input_fallback(self):
        rows = [{'source': 'math', 'query_hash': 'same', 'ppl': 1.0},
                {'source': 'math', 'query_hash': 'same', 'ppl': 9.0},
                {'source': 'math', 'query_hash': 'another', 'ppl': 2.0}]
        self.assertEqual(keep_indices(rows, {('math', 'same')}), [2])

    def test_separate_document_grounded_conditions_are_not_removed_by_source_alias_hold(self):
        rows = [{'source': 'plain', 'query_hash': 'q'}, {'source': 'document', 'query_hash': 'q'}]
        self.assertEqual(keep_indices(rows, {('plain', 'q')}), [1])


if __name__ == '__main__': unittest.main()
