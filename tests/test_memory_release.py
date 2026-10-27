import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from scripts import release_memory_sft_v2 as release


class ReleaseTests(unittest.TestCase):
    def test_concurrent_remote_change_stops_before_upload_or_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'publication-manifest-v1.json').write_text('[]')
            api=MagicMock();api.repo_info.return_value.sha='different-parent'
            with patch.object(release,'STATE',p),patch.object(release,'HfApi',return_value=api):
                with self.assertRaisesRegex(ValueError,'remote head changed'):release.publish()
            api.list_repo_files.assert_not_called();api.create_commit.assert_not_called()

    def test_changed_local_bytes_stop_before_remote_lookup(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'data.parquet').write_bytes(b'changed')
            (p/'publication-manifest-v1.json').write_text(json.dumps([{'path':'data.parquet','sha256':'expected'}]))
            api=MagicMock()
            with patch.object(release,'STATE',p),patch.object(release,'PUBLIC',p),patch.object(release,'HfApi',return_value=api):
                with self.assertRaisesRegex(ValueError,'manifest changed'):release.publish()
            api.repo_info.assert_not_called();api.create_commit.assert_not_called()

    def test_public_readback_is_anonymous_and_hash_bound(self):
        payload=b'public verified bytes';item={'path':'data/x.parquet','sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
        session=MagicMock();session.__enter__.return_value=session
        response=MagicMock();response.__enter__.return_value=response;response.iter_content.return_value=[payload]
        session.get.return_value=response
        with patch.object(release.requests,'Session',return_value=session):
            result=release.anonymous_file(item,'pinned')
        self.assertFalse(session.trust_env);self.assertTrue(result['anonymous'])
        self.assertIn('/resolve/pinned/',session.get.call_args.args[0])
        self.assertNotIn('headers',session.get.call_args.kwargs)

    def test_wrong_remote_bytes_never_become_success(self):
        session=MagicMock();session.__enter__.return_value=session
        response=MagicMock();response.__enter__.return_value=response;response.iter_content.return_value=[b'wrong']
        session.get.return_value=response
        with patch.object(release.requests,'Session',return_value=session),patch.object(release.time,'sleep'):
            with self.assertRaisesRegex(ValueError,'public bytes differ'):
                release.anonymous_file({'path':'x','sha256':'expected','bytes':5},'pinned')


if __name__=='__main__':unittest.main()
