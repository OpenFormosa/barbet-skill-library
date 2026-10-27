"""Pinned inert Dolci bytes for provenance/identity inventory, not admission.

ODC-BY covers database rights. This download does not authorize individual
content licenses, source-level rehosting, or acceptance of every mixture part.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from scripts import build_memory_sft_v2 as base
from scripts.fetch_memory_scale_sources import fetch

ROOT = base.RAW / 'scale10m-dolci'
META = base.RAW / 'scale10m-reference/allenai--Dolci-Instruct-SFT.current-20261027.api.json'
REVISION = 'bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221'


def main():
    metadata = json.loads(META.read_text())
    if metadata['sha'] != REVISION or metadata.get('gated') or metadata.get('cardData', {}).get('license') != 'odc-by':
        raise ValueError('pinned source inventory contract changed')
    files = [f for f in metadata['siblings'] if f['rfilename'].startswith('data/train-') and f['rfilename'].endswith('.parquet')]
    if len(files) != 15: raise ValueError('unexpected source train-file coverage')
    ROOT.mkdir(exist_ok=True)
    if (ROOT / 'lock.json').exists(): raise ValueError('Dolci inventory download already complete')
    jobs = [{'repo': 'allenai/Dolci-Instruct-SFT', 'revision': REVISION, 'path': f['rfilename'],
             'bytes': f['size'], 'local': str(ROOT / f['rfilename'])} for f in files]
    existing = sum(p.stat().st_size for p in base.RAW.rglob('*') if p.is_file())
    needed = sum(j['bytes'] for j in jobs if not Path(j['local']).exists())
    if existing + needed > 128 * 1024 ** 3 or shutil.disk_usage(base.WORK).free - needed < 100 * 1024 ** 3:
        raise ValueError('new raw cache or reserved free disk bound exceeded')
    plan = {'repo': 'allenai/Dolci-Instruct-SFT', 'revision': REVISION, 'metadata_sha256': base.file_sha(META),
        'files': len(jobs), 'bytes': sum(j['bytes'] for j in jobs), 'max_http_workers': 4,
        'source_programs_executed': False, 'accepted_training_rows': 0,
        'purpose': 'Per-source provenance, license, distinct query, context and tool-schema inventory. No blanket source admission.'}
    plan_path = ROOT / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan: raise ValueError('source plan changed')
    if not plan_path.exists(): base.save(plan_path, plan)
    expected = {f['rfilename']: f for f in files}; results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, result in enumerate(pool.map(fetch, jobs), 1):
            upstream = expected[result['path']]
            if result['sha256'] != upstream['lfs']['sha256'] or result['bytes'] != upstream['size']:
                raise ValueError('source bytes do not match pinned primary LFS hash')
            results.append(result); print(json.dumps({'downloaded': n, 'total': len(jobs)}), flush=True)
    base.save(ROOT / 'lock.json', results)
    print(json.dumps({'verified_files': len(results), 'verified_bytes': sum(r['bytes'] for r in results), 'accepted_training_rows': 0}), flush=True)


if __name__ == '__main__': main()
