import unittest
from scripts.recover_v26_publication_metadata_v1 import metadata_only_paths


class MetadataRecoveryTests(unittest.TestCase):
    def test_training_payload_gap_cannot_be_called_metadata_repair(self):
        with self.assertRaisesRegex(ValueError, 'training payload gap'):
            metadata_only_paths(['data/wave-0003/records/train-00000.parquet'])

    def test_only_the_observed_two_file_gap_is_admitted(self):
        self.assertEqual(len(metadata_only_paths(['recipe/memory_sft_v2_scale_10m.json', 'recipe/memory_sft_v2.json'])), 2)
        with self.assertRaisesRegex(ValueError, 'unexpected missing'):
            metadata_only_paths(['README.md'])


if __name__ == '__main__': unittest.main()
