import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import recover_v26_overlap_candidates as recovery


class OverlapRecoveryTests(unittest.TestCase):
    def test_only_current_wave_query_dedup_is_removed_legacy_and_first_holds_remain(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); first = root / 'first'; score = root / 'score'
            first.mkdir(); score.mkdir()
            (first / 'exclusions.private.json').write_text(json.dumps({'old_query_hashes': ['legacy']}))
            (score / 'candidate-holds.private.json').write_text(json.dumps({
                'query_hashes': ['negative-review'], 'supplemental_prior_queries': ['prior-job']}))
            (score / 'candidates.sqlite').touch()
            with patch.object(recovery, 'FIRST', first), patch.object(recovery.candidates, 'STATE', score), \
                 patch.object(recovery.wave, 'init'), patch.object(recovery.candidates, 'init'), \
                 patch.object(recovery.base, 'EX', {'old_query_hashes': {'current-wave', 'legacy'}}, create=True):
                recovery.init()
                self.assertEqual(recovery.base.EX['old_query_hashes'], {'legacy', 'negative-review', 'prior-job'})
                recovery.EXISTING.close()


if __name__ == '__main__':
    unittest.main()
