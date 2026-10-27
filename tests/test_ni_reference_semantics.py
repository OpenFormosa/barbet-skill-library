import json
import unittest
from scripts.package_natural_instructions_scope_v2 import CONFIG, scope_hold


class ReferenceSemanticsTest(unittest.TestCase):
    def setUp(self): self.policy = json.loads(CONFIG.read_text())

    def check(self, name, alternatives=2):
        return scope_hold({'file': 'tasks/' + name, 'reference_alternatives': alternatives}, self.policy)

    def test_required_entities_are_not_answer_versions(self):
        for name in ['task002_quoref_answer_generation.json', 'task390_torque_text_span_selection.json',
                     'task1518_limit_answer_generation.json']:
            self.assertEqual(self.check(name), 'joint_or_unresolved_reference_output_semantics')

    def test_complete_decomposition_or_paraphrase_versions_still_rank(self):
        self.assertIsNone(self.check('task168_strategyqa_question_decomposition.json'))
        self.assertIsNone(self.check('task111_asset_sentence_simplification.json'))
        self.assertEqual(self.check('task9999_unknown_generation.json'), 'unreviewed_multi_reference_contract')

    def test_background_examples_do_not_supply_missing_target_facts(self):
        self.assertEqual(self.check('task344_hybridqa_answer_generation.json', 1), 'target_facts_not_supplied')
        self.assertEqual(self.check('task672_nummersense.json', 1), 'target_facts_not_supplied')
        self.assertIsNone(self.check('task672_amazon_and_yelp_summarization_dataset_summarization.json'))
        self.assertIsNone(self.check('task345_hybridqa_answer_generation.json', 1))

    def test_official_ni_train_is_not_original_benchmark_split_proof(self):
        self.assertEqual(self.check('task664_mmmlu_answer_generation_abstract_algebra.json', 1),
                         'source_benchmark_without_verified_upstream_train_provenance')


if __name__ == '__main__': unittest.main()
