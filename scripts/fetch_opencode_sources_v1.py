"""Fetch immutable, openly licensed OpenCodeInstruct as inert source bytes."""
from concurrent.futures import ThreadPoolExecutor
import json
import shutil

from scripts import build_memory_sft_v2 as base
from scripts import fetch_memory_sft_v2 as fetcher

STATE = base.RAW / 'scale10m-opencode'
METADATA = base.RAW / 'scale10m-reference/nvidia--OpenCodeInstruct.metadata-20261027.json'
REVISION = '8f3ba5bafe4d6e8db46082cf7ae6741bc370604d'


def main():
    info = json.loads(METADATA.read_text())
    if info['revision'] != REVISION or info['gated'] or info['card']['license'] != 'cc-by-4.0':
        raise ValueError('pinned public-source contract changed')
    files = [f for f in info['files'] if f['path'].startswith('data/train-') and f['path'].endswith('.parquet')]
    if len(files) != 50: raise ValueError('unexpected primary train file coverage')
    needed = sum(f['bytes'] for f in files)
    present = sum(p.stat().st_size for p in base.RAW.rglob('*') if p.is_file())
    if present + needed > 128 * 1024**3 or shutil.disk_usage(base.WORK).free - needed < 100 * 1024**3:
        raise ValueError('bounded raw cache or free disk budget exceeded')
    STATE.mkdir(exist_ok=True)
    if (STATE / 'lock.json').exists(): raise ValueError('source lock already complete')
    base.save(STATE / 'prospective-fetch-v1.json', {'repo': info['repo'], 'revision': REVISION,
        'files': files, 'prospective_bytes': needed, 'metadata_sha256': base.file_sha(METADATA),
        'maximum_http_workers': 4, 'source_programs_executed': False, 'accepted_training_rows': 0})
    fetcher.OUT = STATE
    jobs = [(info['repo'], REVISION, f['path'], f['bytes']) for f in files]
    with ThreadPoolExecutor(max_workers=4) as pool: results = list(pool.map(fetcher.fetch, jobs))
    expected = {f['path']: f for f in files}
    for item in results:
        remote = expected[item['path']]
        if item['bytes'] != remote['bytes'] or item['sha256'] != remote['lfs']['sha256']:
            raise ValueError('downloaded data differs from pinned primary LFS hash')
    base.save(STATE / 'lock.json', results)
    print(json.dumps({'files': len(results), 'bytes': sum(r['bytes'] for r in results),
                      'all_primary_hashes_verified': True, 'accepted_training_rows': 0}), flush=True)


if __name__ == '__main__': main()
