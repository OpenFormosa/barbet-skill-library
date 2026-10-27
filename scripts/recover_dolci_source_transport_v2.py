"""Bounded public-byte retry after an inert source download stalls.

The original partial file is preserved. Complete files are hash-verified, not
downloaded again. This never retries a model generation or changes source scope.
"""
import hashlib
import json
from pathlib import Path
import time
import uuid

import requests
from scripts import build_memory_sft_v2 as base
from scripts import fetch_dolci_inventory_v1 as original


def bounded_fetch(entry):
    dest = Path(entry['local'])
    expected = next(f for f in json.loads(original.META.read_text())['siblings'] if f['rfilename'] == entry['path'])['lfs']['sha256']
    if dest.exists():
        if dest.stat().st_size != entry['bytes'] or base.file_sha(dest) != expected:
            raise ValueError('existing complete source differs from primary hash')
        return {**entry, 'sha256': expected}
    for attempt in range(1, 4):
        receipt = original.ROOT / f'transport-v2-{base.sha(entry["path"])}-{attempt}.json'
        partial = dest.with_suffix(f'.transport-v2-{attempt}.partial')
        if receipt.exists() or partial.exists(): raise ValueError('prior retry evidence exists; inspect before resuming')
        start = time.monotonic(); digest = hashlib.sha256(); size = 0
        url = f'https://huggingface.co/datasets/{entry["repo"]}/resolve/{entry["revision"]}/{entry["path"]}?download=true&transport_attempt={uuid.uuid4().hex}'
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.get(url, stream=True, timeout=(15, 30)) as response:
                    response.raise_for_status()
                    with partial.open('wb') as handle:
                        for chunk in response.iter_content(64 * 1024):
                            if time.monotonic() - start > 180: raise TimeoutError('source transfer wall deadline')
                            size += len(chunk)
                            if size > entry['bytes']: raise ValueError('unexpected source size')
                            digest.update(chunk); handle.write(chunk)
            if size != entry['bytes'] or digest.hexdigest() != expected: raise ValueError('source size or primary SHA256 mismatch')
            partial.replace(dest)
            base.save(receipt, {'path': entry['path'], 'bytes': size, 'sha256': digest.hexdigest(),
                'seconds': time.monotonic() - start, 'success': True, 'anonymous': True})
            return {**entry, 'sha256': expected}
        except Exception as exc:
            base.save(receipt, {'path': entry['path'], 'bytes': size, 'partial_sha256': digest.hexdigest(),
                'seconds': time.monotonic() - start, 'success': False, 'error_type': type(exc).__name__,
                'partial_bytes_preserved': True})
            if attempt == 3: raise


if __name__ == '__main__':
    failure = original.ROOT / 'first-download-stall-v1.json'
    if not failure.exists(): raise ValueError('first download evidence required')
    binding = original.ROOT / 'bounded-transport-v2.json'
    if binding.exists(): raise ValueError('bounded recovery already launched')
    base.save(binding, {'driver_sha256': base.file_sha(Path(__file__)),
        'original_downloader_sha256': base.file_sha(Path(original.__file__)),
        'original_failure_sha256': base.file_sha(failure), 'plan_sha256': base.file_sha(original.ROOT / 'plan.json'),
        'max_attempts': 3, 'chunk_bytes': 65536, 'wall_deadline_seconds': 180,
        'source_scope_unchanged': True, 'original_partial_bytes_retained': True,
        'source_programs_executed': False, 'model_requests': 0})
    original.fetch = bounded_fetch; original.main()
