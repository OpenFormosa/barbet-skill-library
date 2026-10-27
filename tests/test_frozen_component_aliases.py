import unittest
from scripts.freeze_v26_release_folds_v1 import Components, add_frozen


class FrozenComponentTests(unittest.TestCase):
    def test_losing_alternative_alias_propagates_frozen_fold(self):
        graph = Components()
        graph.union('family:winner', 'input:same-condition')
        graph.union('family:loser', 'input:same-condition')
        graph.union('family:later-source', 'family:loser')
        self.assertEqual(graph.find('family:later-source'), graph.find('family:winner'))

    def test_cross_fold_bridge_is_a_hold_not_a_reassignment(self):
        ties = {}; conflicts = set()
        add_frozen(ties, conflicts, 'component', 'family-a', 'train')
        add_frozen(ties, conflicts, 'component', 'family-b', 'validation')
        self.assertEqual(ties['component'], ('family-a', 'train'))
        self.assertIn('component', conflicts)


if __name__ == '__main__': unittest.main()
