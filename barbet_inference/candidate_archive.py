"""Lossless, independently checksummed records for new candidate cohorts.

Each frame contains one exact UTF-8 JSON payload. Random access needs no model,
source adapter, external compression dependency, or full-shard decompression.
Existing JSONL cohorts and their frozen readers are not changed.
"""
import hashlib
from pathlib import Path
import struct
import zlib

FORMAT = 'barbet-candidate-zlib-frames-v1'
MAGIC = b'BARBET-CANDIDATES\x00v1\n'
HEADER = struct.Struct('<QQ32s')
MAX_RAW_BYTES = 16 * 1024 * 1024
MAX_COMPRESSED_BYTES = MAX_RAW_BYTES + 65536


class ArchiveError(ValueError):
    pass


def _magic(handle):
    if handle.read(len(MAGIC)) != MAGIC:
        raise ArchiveError('candidate archive magic differs')


class ArchiveWriter:
    def __init__(self, path):
        self.handle = Path(path).open('xb')
        self.handle.write(MAGIC)
        self.records = self.raw_bytes = 0

    def append(self, raw):
        if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_RAW_BYTES:
            raise ArchiveError('candidate raw payload outside bounded frame scope')
        raw.decode('utf-8')
        compressed = zlib.compress(raw, level=6)
        if len(compressed) > MAX_COMPRESSED_BYTES:
            raise ArchiveError('compressed frame byte bound')
        offset = self.handle.tell()
        self.handle.write(HEADER.pack(len(compressed), len(raw), hashlib.sha256(raw).digest()))
        self.handle.write(compressed)
        self.records += 1
        self.raw_bytes += len(raw)
        return offset, HEADER.size + len(compressed)

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def _frame(handle, expected_length=None):
    header = handle.read(HEADER.size)
    if not header:
        return None
    if len(header) != HEADER.size:
        raise ArchiveError('truncated candidate frame header')
    compressed_size, raw_size, digest = HEADER.unpack(header)
    if not 0 < raw_size <= MAX_RAW_BYTES or not 0 < compressed_size <= MAX_COMPRESSED_BYTES:
        raise ArchiveError('candidate frame size bound')
    length = HEADER.size + compressed_size
    if expected_length is not None and length != expected_length:
        raise ArchiveError('candidate pointer length differs')
    data = handle.read(compressed_size)
    if len(data) != compressed_size:
        raise ArchiveError('truncated candidate frame payload')
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(data, raw_size + 1)
    except zlib.error as exc:
        raise ArchiveError('candidate decompression failed') from exc
    if len(raw) != raw_size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ArchiveError('candidate frame decoded size or stream differs')
    if hashlib.sha256(raw).digest() != digest:
        raise ArchiveError('candidate raw checksum differs')
    raw.decode('utf-8')
    return length, raw


def iter_records(path):
    with Path(path).open('rb') as handle:
        _magic(handle)
        while True:
            offset = handle.tell()
            frame = _frame(handle)
            if frame is None:
                break
            length, raw = frame
            yield offset, length, raw


def read_record(path, offset, length):
    if not isinstance(offset, int) or not isinstance(length, int) or offset < len(MAGIC) or length <= HEADER.size:
        raise ArchiveError('invalid candidate archive pointer')
    with Path(path).open('rb') as handle:
        _magic(handle)
        handle.seek(offset)
        frame = _frame(handle, length)
    if frame is None:
        raise ArchiveError('candidate pointer reaches EOF')
    return frame[1]


def load_candidate(pointer, files):
    """Explicit prospective loader; canonical identity logic stays unchanged."""
    import json
    from scripts.prepare_v26_answer_candidates import canonicalize
    spec = files[pointer['file_id']]
    if spec.get('format') != FORMAT:
        raise ArchiveError('candidate manifest does not declare framed storage')
    raw = read_record(spec['path'], pointer['offset'], pointer['length'])
    row = canonicalize(json.loads(raw))
    if (row['candidate_identity'] != pointer['candidate_identity']
            or row['ppl_input_hash'] != pointer['input_hash']
            or row['record_id'] != pointer['record_id']):
        raise ArchiveError('candidate pointer identity binding differs')
    return row
