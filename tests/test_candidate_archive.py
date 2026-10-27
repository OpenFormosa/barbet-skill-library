import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zlib

from barbet_inference import candidate_archive as archive
from barbet_inference.memory import canonical
from scripts.prepare_v26_answer_candidates import canonicalize, load_candidate as load_jsonl


class CandidateArchiveTests(unittest.TestCase):
    def test_literal_payload_and_random_access_are_exact(self):
        payloads = [b'{"content":"  A \\n \\t  ","n":-2}',
                    canonical({'content': '臺灣。\\n真實\n換行', 'loss': False}).encode()]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rows.bca'
            with archive.ArchiveWriter(path) as writer:
                pointers = [writer.append(raw) for raw in payloads]
            self.assertEqual([x[2] for x in archive.iter_records(path)], payloads)
            for index in [1, 0]:
                self.assertEqual(archive.read_record(path, *pointers[index]), payloads[index])
            with self.assertRaises(FileExistsError):
                archive.ArchiveWriter(path)

    def test_truncation_checksum_and_pointer_mismatch_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rows.bca'
            with archive.ArchiveWriter(path) as writer:
                offset, length = writer.append(b'{"answer":17}')
            original = path.read_bytes()
            for damaged in [original[:-1], original[:offset+8],
                            original[:offset+16] + b'x' * 32 + original[offset+48:]]:
                path.write_bytes(damaged)
                with self.assertRaises(archive.ArchiveError):
                    list(archive.iter_records(path))
            path.write_bytes(original)
            with self.assertRaises(archive.ArchiveError):
                archive.read_record(path, offset, length + 1)

    def test_extra_stream_and_decompression_size_lie_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rows.bca'
            raw = b'A' * 1024
            for compressed, declared in [(zlib.compress(raw) + zlib.compress(b'B'), len(raw)),
                                         (zlib.compress(raw), 10)]:
                path.write_bytes(archive.MAGIC + archive.HEADER.pack(len(compressed), declared, hashlib.sha256(raw).digest()) + compressed)
                with self.assertRaises(archive.ArchiveError):
                    list(archive.iter_records(path))

    def test_old_and_new_loaders_preserve_every_canonical_field(self):
        row = {'record_id': 'before-canonicalization', 'family_id': 'family', 'query_hash': 'query',
               'source': 'test-fixture', 'source_refs': {}, 'tools': [],
               'messages': [{'role': 'user', 'content': '  Evaluate -2^3 - (-3)^2.\n', 'loss': False},
                            {'role': 'assistant', 'content': '-17', 'loss': True}]}
        expected = canonicalize(copy.deepcopy(row))
        pointer = {'file_id': 0, 'candidate_identity': expected['candidate_identity'],
                   'input_hash': expected['ppl_input_hash'], 'record_id': expected['record_id']}
        raw = (canonical(row) + '\n').encode()
        with tempfile.TemporaryDirectory() as folder:
            plain = Path(folder) / 'rows.jsonl'; plain.write_bytes(raw)
            framed = Path(folder) / 'rows.bca'
            with archive.ArchiveWriter(framed) as writer:
                offset, length = writer.append(raw)
            a = load_jsonl({**pointer, 'offset': 0, 'length': len(raw)}, [{'path': str(plain)}])
            b = archive.load_candidate({**pointer, 'offset': offset, 'length': length},
                                       [{'path': str(framed), 'format': archive.FORMAT}])
            self.assertEqual(a, b)
            self.assertEqual(a, expected)
            with self.assertRaises(archive.ArchiveError):
                archive.load_candidate({**pointer, 'offset': offset, 'length': length, 'record_id': 'different'},
                                       [{'path': str(framed), 'format': archive.FORMAT}])


if __name__ == '__main__':
    unittest.main()
