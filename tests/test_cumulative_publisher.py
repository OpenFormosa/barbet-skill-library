import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch
from scripts import publish_memory_cumulative_v2 as publisher


class CumulativePublisherTests(unittest.TestCase):
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
