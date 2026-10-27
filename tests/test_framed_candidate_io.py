import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from barbet_inference import candidate_archive as archive
from barbet_inference.memory import canonical
from scripts import build_memory_sft_v2 as base
from scripts import prepare_v26_answer_candidates as candidates
from scripts import guard_v26_episode_identity as identity
from scripts import framed_candidate_io_v1 as framed


class FramedCandidateIOTests(unittest.TestCase):
    def test_prefix_holds_observation_ambiguity_and_alternatives_match_old_pipeline(self):
        tool = {'type': 'function', 'function': {'name': 'lookup', 'description': 'Return the value.',
                'parameters': {'type': 'object', 'properties': {}}}}
        messages = [{'role': 'user', 'content': 'Look up the value.', 'loss': False},
                    {'role': 'assistant', 'content': '', 'loss': True,
                     'tool_calls': [{'id': 'original-call', 'type': 'function',
                                     'function': {'name': 'lookup', 'arguments': {}}}]}]
        template = {'record_id': 'original', 'family_id': 'tool-family', 'query_hash': 'tool-query',
                    'source': 'inert-test-fixture', 'source_refs': {}, 'domain': 'tool_use', 'tools': [tool]}
        rows = [{**copy.deepcopy(template), 'messages': copy.deepcopy(messages)}]
        for value in ['2', '3']:
            rows.append({**copy.deepcopy(template), 'messages': copy.deepcopy(messages) + [
                {'role': 'tool', 'name': 'lookup', 'tool_call_id': 'original-call', 'content': value, 'loss': False},
                {'role': 'assistant', 'content': value, 'loss': True}]})
        for answer in ['5', '2 + 3 = 5.']:
            rows.append({**copy.deepcopy(template), 'domain': 'math', 'tools': [], 'family_id': 'math-family',
                         'query_hash': 'math-query', 'messages': [
                             {'role': 'user', 'content': 'Compute 2 + 3.', 'loss': False},
                             {'role': 'assistant', 'content': answer, 'loss': True}]})
        old_state, old_identity = candidates.STATE, identity.STATE
        old_index, old_guard = candidates.index_file, identity.job
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder); plain = root / 'fixture.jsonl'; packed = root / 'fixture.bca'
                raw = [(canonical(row) + '\n').encode() for row in rows]
                plain.write_bytes(b''.join(raw))
                with archive.ArchiveWriter(packed) as writer:
                    for payload in raw:
                        writer.append(payload)
                result = {}
                for label, path in [('plain', plain), ('framed', packed)]:
                    state = root / label; state.mkdir()
                    spec = {'file_id': 0, 'path': str(path), 'sha256': base.file_sha(path)}
                    if label == 'framed':
                        spec['format'] = archive.FORMAT
                    base.save(state / 'candidate-input-files.json', [spec])
                    base.save(state / 'candidate-holds.private.json', {'record_ids': [], 'query_hashes': [],
                              'search_allowed': [], 'supplemental_prior_queries': []})
                    candidates.STATE = state; candidates.index_file = old_index if label == 'plain' else framed.index_file
                    candidates.index(1)
                    identity.STATE = state / 'episode-identity-v1'; identity.job = old_guard if label == 'plain' else framed.guard_file
                    identity.main()
                    db = sqlite3.connect(identity.STATE / 'identity.sqlite')
                    result[label] = db.execute('SELECT * FROM holds ORDER BY candidate_identity').fetchall(); db.close()
                    summary = json.loads((state / 'candidate-index-summary.json').read_text())
                    self.assertEqual(summary['distinct_answers'], 5)
                    self.assertEqual(summary['competing_groups'], 1)
                self.assertEqual(result['plain'], result['framed'])
                reasons = [r[1] for r in result['framed']]
                self.assertEqual(reasons.count('strict_dialogue_prefix_view'), 1)
                self.assertEqual(reasons.count('same_task_different_observation_conditions'), 2)
        finally:
            candidates.STATE = old_state; identity.STATE = old_identity
            candidates.index_file = old_index; identity.job = old_guard


if __name__ == '__main__':
    unittest.main()
