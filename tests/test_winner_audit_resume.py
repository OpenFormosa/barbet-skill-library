import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import resume_v26_winner_audit_v1 as resume


class WinnerAuditResumeTests(unittest.TestCase):
    JOB = {'id': 'sample-abc', 'record_id': 'abc', 'domain': 'coding'}

    def test_first_incomplete_request_is_held_without_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp); folder = state / 'semantic-audit-v1/sample-abc'; folder.mkdir(parents=True)
            request = folder / 'request.json'; request.write_text('{"original":true}\n')
            with patch.object(resume, 'STATE', state), patch.object(resume.reviewer, 'review') as call:
                result = resume.resume_one(self.JOB)
            call.assert_not_called(); self.assertEqual(result['verdict'], 'HOLD')
            self.assertEqual(request.read_text(), '{"original":true}\n')
            self.assertFalse((folder / 'receipt.json').exists())

    def test_old_negative_verdict_is_reused_and_new_job_gets_its_first_call(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp); folder = state / 'semantic-audit-v1/sample-abc'; folder.mkdir(parents=True)
            (folder / 'disposition.json').write_text(json.dumps({'verdict': 'REJECT', 'first': True}))
            with patch.object(resume, 'STATE', state), patch.object(resume.reviewer, 'review', return_value={'verdict': 'ACCEPT'}) as call:
                self.assertEqual(resume.resume_one(self.JOB), {'verdict': 'REJECT', 'first': True})
                call.assert_not_called()
                new = {**self.JOB, 'id': 'sample-new', 'record_id': 'new'}
                self.assertEqual(resume.resume_one(new), {'verdict': 'ACCEPT'})
                call.assert_called_once_with(new)


if __name__ == '__main__': unittest.main()
