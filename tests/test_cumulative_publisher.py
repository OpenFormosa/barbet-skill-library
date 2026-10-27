import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch
from scripts import publish_memory_cumulative_v2 as publisher


class CumulativePublisherTests(unittest.TestCase):
    def test_shared_recipe_folder_survives_sdk_unchanged_add_optimization(self):
        existing={'.gitattributes','README.md','recipe/shared.json','recipe/old.json','data/wave-1/records/x.parquet'}
        planned={'README.md','recipe/shared.json','recipe/new.json','data/wave-3/records/y.parquet'}
        operations=publisher.replacement_deletions(existing,planned)
        tree=set(existing)
        for op in operations:
            tree={p for p in tree if not (p==op.path_in_repo or (op.is_folder and p.startswith(op.path_in_repo.rstrip('/')+'/')))}
        # The SDK drops the unchanged Add for recipe/shared.json. Its old blob
        # must remain reachable because no ancestor was deleted.
        tree.update(planned-{'recipe/shared.json'})
        self.assertEqual(tree,planned|{'.gitattributes'})

    def test_wholly_obsolete_subtree_is_deleted_without_one_operation_per_shard(self):
        ops=publisher.replacement_deletions({'data/wave-1/a','data/wave-1/b','data/wave-2/c'}, {'data/wave-3/d'})
        self.assertEqual({(o.path_in_repo.rstrip('/'),o.is_folder) for o in ops},{('data/wave-1',True),('data/wave-2',True)})

    def test_user_supersession_blocks_publication_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'SUPERSEDED.json').write_text('{}')
            api=MagicMock()
            with patch.object(publisher,'STATE',p),patch.object(publisher,'HfApi',return_value=api):
                for action in [publisher.prepare,publisher.publish]:
                    with self.assertRaisesRegex(ValueError,'superseded'):action()
            api.create_commit.assert_not_called();api.repo_info.assert_not_called()

    def test_changed_remote_parent_is_never_deleted(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'publication-plan.json').write_text(json.dumps({'parent_commit':'old','manifest':[]}))
            api=MagicMock();api.repo_info.return_value.sha='new'
            with patch.object(publisher,'STATE',p),patch.object(publisher,'HfApi',return_value=api):
                with self.assertRaisesRegex(ValueError,'remote changed'):publisher.publish()
            api.create_commit.assert_not_called();api.list_repo_files.assert_not_called()

    def test_changed_local_file_stops_before_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'x').write_text('changed')
            (p/'publication-plan.json').write_text(json.dumps({'parent_commit':'old','manifest':[{'path':'x','sha256':'wrong'}]}))
            api=MagicMock();api.repo_info.return_value.sha='old'
            with patch.object(publisher,'STATE',p),patch.object(publisher,'DEST',p),patch.object(publisher,'HfApi',return_value=api):
                with self.assertRaisesRegex(ValueError,'planned bytes'):publisher.publish()
            api.create_commit.assert_not_called();api.list_repo_files.assert_not_called()


if __name__=='__main__':unittest.main()
