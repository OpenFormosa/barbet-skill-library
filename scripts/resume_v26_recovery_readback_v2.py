"""Retry stalled anonymous transport with bounded per-file wall time.

Only the same frozen revision/manifest may produce receipts. Earlier successful
byte receipts remain unchanged. This makes no model request or data publication.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time
from urllib.parse import quote

import requests
from scripts import build_memory_sft_v2 as base
from scripts.recover_v26_publication_metadata_v1 import STATE
from scripts.publish_memory_cumulative_v2 import DATASET


def verify(item, revision, folder):
    expected = {**item, 'revision': revision, 'anonymous': True}
    receipt = folder / (hashlib.sha256(item['path'].encode()).hexdigest() + '.json')
    if receipt.exists():
        if json.loads(receipt.read_text()) != expected: raise ValueError('receipt binding conflict')
        return
    url = f'https://huggingface.co/datasets/{DATASET}/resolve/{revision}/' + quote(item['path'], safe='/')
    attempts = []
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(3):
            started = time.monotonic(); count = 0; digest = hashlib.sha256()
            try:
                # A fresh anonymous redirect avoids persisting a stalled CDN
                # connection. The immutable commit still fixes all file bytes.
                with session.get(url, params={'download': 'true', 'readback_attempt': str(time.time_ns())},
                                 stream=True, timeout=(20, 30), headers={'Cache-Control': 'no-cache'}) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(64 * 1024):
                        count += len(chunk); digest.update(chunk)
                        if time.monotonic() - started > 180: raise TimeoutError('anonymous file wall-time bound')
                if count != item['bytes'] or digest.hexdigest() != item['sha256']:
                    raise ValueError('anonymous full-byte checksum mismatch')
                temporary = receipt.with_suffix('.retry-v2.partial')
                temporary.write_text(json.dumps(expected, sort_keys=True) + '\n'); os.replace(temporary, receipt)
                attempts.append({'attempt': attempt, 'bytes': count, 'seconds': time.monotonic() - started, 'verified': True})
                base.save(STATE / 'anonymous-readback/transport-retry-v2' / receipt.name,
                          {'expected': expected, 'attempts': attempts, 'actual_full_bytes_read': True})
                print(json.dumps({'verified': item['path'], 'bytes': count, 'seconds': time.monotonic() - started}), flush=True)
                return
            except (requests.RequestException, TimeoutError, ValueError) as exc:
                attempts.append({'attempt': attempt, 'bytes': count, 'seconds': time.monotonic() - started, 'error': type(exc).__name__})
                print(json.dumps({'retry': item['path'], **attempts[-1]}), flush=True)
                if attempt == 2:
                    base.save(STATE / 'anonymous-readback/transport-retry-v2' / receipt.name,
                              {'expected': expected, 'attempts': attempts, 'actual_full_bytes_read': False})
                    raise


if __name__ == '__main__':
    plan = json.loads((STATE / 'publication-plan.json').read_text())
    data = json.loads((STATE / 'publication-data-commit.json').read_text())
    if data['plan_sha256'] != base.file_sha(STATE / 'publication-plan.json'): raise ValueError('plan changed')
    folder = STATE / 'anonymous-readback/files'
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: verify(item, data['commit'], folder), plan['manifest']))
