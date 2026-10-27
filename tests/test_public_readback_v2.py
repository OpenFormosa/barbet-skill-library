import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import verify_memory_publication_v2 as verify


class ReadbackTests(unittest.TestCase):
    def test_bound_receipt_resumes_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);item={'path':'data/file','sha256':'hash','bytes':9}
            receipt=p/(hashlib.sha256(item['path'].encode()).hexdigest()+'.json')
            receipt.write_text(json.dumps({**item,'revision':'pinned','anonymous':True}))
            with patch.object(verify.requests,'Session') as session:
                result=verify.verify_file(item,'pinned',p)
            session.assert_not_called();self.assertEqual(result['revision'],'pinned')

    def test_receipt_for_different_bytes_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);item={'path':'data/file','sha256':'new','bytes':9}
            receipt=p/(hashlib.sha256(item['path'].encode()).hexdigest()+'.json')
            receipt.write_text(json.dumps({**item,'sha256':'old','revision':'pinned','anonymous':True}))
            with self.assertRaisesRegex(ValueError,'conflict'):verify.verify_file(item,'pinned',p)


if __name__=='__main__':unittest.main()
