import copy
import unittest

from scripts.extend_v26_overlap_selection import protocol_hash, rebind_score


class FrozenScoreReuseTests(unittest.TestCase):
    def binding(self, manifest):
        return {'policy_sha256': 'p', 'scorer_sha256': 's', 'runner_sha256': 'r',
                'adapter_sha256': 'a', 'canonical_batch_size': 1, 'model_revision': 'v26',
                'candidate_manifest_sha256': manifest, 'candidate_summary_sha256': manifest,
                'bucket': 0, 'limit': 0}

    def test_extension_preserves_every_numerical_value_and_original_object(self):
        old, new = self.binding('original'), self.binding('extended')
        score = {'protocol_sha256': protocol_hash(old), 'mean_nll': 1.2345,
                 'ppl': 3.436659761170463, 'answer_tokens': 13, 'sequence_tokens': 100,
                 'text_sha256': 'exact-native-text', 'record_id': 'candidate'}
        before = copy.deepcopy(score)
        result = rebind_score(score, old, new, 'old-database')
        self.assertEqual(score, before)
        for key in ['mean_nll', 'ppl', 'answer_tokens', 'sequence_tokens', 'text_sha256', 'record_id']:
            self.assertEqual(result[key], score[key])
        self.assertEqual(result['protocol_sha256'], protocol_hash(new))
        self.assertEqual(result['reused_from']['protocol_sha256'], score['protocol_sha256'])

    def test_changed_model_mask_renderer_or_batch_protocol_cannot_reuse_scores(self):
        old = self.binding('original'); score = {'protocol_sha256': protocol_hash(old)}
        for key in ['policy_sha256', 'scorer_sha256', 'runner_sha256', 'adapter_sha256',
                    'canonical_batch_size', 'model_revision']:
            new = self.binding('extended'); new[key] = 'different'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'numerical reuse protocol changed'):
                rebind_score(score, old, new, 'db')

    def test_score_with_an_unbound_original_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'original protocol'):
            rebind_score({'protocol_sha256': 'unrelated'}, self.binding('old'), self.binding('new'), 'db')


if __name__ == '__main__':
    unittest.main()
