import unittest
import pyarrow as pa
from scripts.publish_source_contract_pruned_v1 import filter_aligned_views


class SourceContractPruningTests(unittest.TestCase):
    def views(self):
        records = pa.table({'record_id': ['bad-schema', 'same-query-other-view', 'unrelated'],
                            'query_hash': ['held-query', 'held-query', 'keep-query'],
                            'text': ['A', 'B', 'C']})
        tokens = pa.table({'record_id': records['record_id'], 'input_ids': [[1, 2], [3, 4], [5, 6, 7]],
                           'labels': [[-100, 2], [-100, 4], [-100, 6, -100]]})
        return {'records': records, 'tokenized': tokens, 'minicpm5_tokenized': tokens}

    def test_entire_question_held_and_remaining_labels_exact(self):
        views = self.views(); selected, removed = filter_aligned_views(views, {'held-query'})
        self.assertEqual(removed, ['bad-schema', 'same-query-other-view'])
        for key, table in selected.items(): self.assertTrue(table.equals(views[key].slice(2)))
        self.assertEqual(selected['tokenized']['labels'].to_pylist(), [[-100, 6, -100]])

    def test_misaligned_token_views_fail_before_filtering(self):
        views = self.views(); views['tokenized'] = views['tokenized'].take(pa.array([1, 0, 2]))
        with self.assertRaisesRegex(ValueError, 'not aligned'): filter_aligned_views(views, {'held-query'})


if __name__ == '__main__': unittest.main()
