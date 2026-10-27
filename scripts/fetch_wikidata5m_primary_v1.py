"""Freeze original-author graph and alias downloads; no training admission.

The Wikipedia prose corpus is deliberately outside this source investigation.
Primary archives have no published checksum on the project page; record the
observed complete SHA256 honestly instead of claiming a reference-hash match.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

import requests
from scripts import build_memory_sft_v2 as base

RAW = base.WORK / 'data/raw_memory_harness_v2/scale10m-wikidata5m'
STATE = base.WORK / 'reports/memory-harness-v2-wikidata5m-v1'
URLS = {
    'wikidata5m_inductive.tar.gz': 'https://www.dropbox.com/s/csed3cgal3m7rzo/wikidata5m_inductive.tar.gz?dl=1',
    'wikidata5m_alias.tar.gz': 'https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1',
}
PAGES = {
    'original-project.html': 'https://deepgraphlearning.github.io/project/wikidata5m',
    'wikidata-structured-license.html': 'https://www.wikidata.org/wiki/Wikidata:Licensing',
}
MAX_FILE_BYTES = 256 * 1024 ** 2


def prepare():
    RAW.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
    plan = STATE / 'source-plan.json'
    if plan.exists():
        value = json.loads(plan.read_text())
        if value['driver_sha256'] != base.file_sha(Path(__file__)):
            raise ValueError('source freeze code changed')
        return
    resources = json.loads((base.REPO / 'configs/memory_sft_v2_scale_10m_resources_v2.json').read_text())['resources']
    raw_bytes = sum(p.stat().st_size for p in RAW.parent.rglob('*') if p.is_file())
    reserve = 2 * MAX_FILE_BYTES + 4 * 1024 ** 2
    if raw_bytes + reserve > resources['new_raw_cache_cap_bytes'] or shutil.disk_usage(RAW).free < resources['minimum_free_disk_bytes'] + reserve:
        raise ValueError('source raw cap or disk floor would be exceeded')
    references = []
    for name, url in PAGES.items():
        response = requests.get(url, timeout=(10, 30)); response.raise_for_status()
        if len(response.content) > 2 * 1024 ** 2:
            raise ValueError('source reference page size bound')
        (RAW / name).write_bytes(response.content)
        references.append({'url': url, 'file': name, 'sha256': base.file_sha(RAW / name), 'bytes': len(response.content)})
    base.save(plan, {'driver_sha256': base.file_sha(Path(__file__)), 'urls': URLS, 'primary_references': references,
        'raw_bytes_before_download': raw_bytes, 'reserved_download_bytes': reserve, 'maximum_file_bytes': MAX_FILE_BYTES,
        'original_source_expected_inductive_train_entities': 4579609,
        'original_source_expected_inductive_train_triples': 20496514,
        'license_scope': 'Wikidata structured triples and entity/property aliases are investigated under official CC0 policy. Wikipedia article prose is not downloaded or relabeled.',
        'split_scope': 'Download original inductive train/valid/test archive for inventory and exclusion bookkeeping; only training facts may become prospective task evidence.',
        'published_upstream_checksum_available': False, 'model_requests': 0, 'accepted_training_rows': 0,
        'source_programs_executed': False, 'created_at': datetime.now(timezone.utc).isoformat()})


def fetch(item):
    name, url = item; path = RAW / name; partial = path.with_suffix(path.suffix + '.partial'); receipt = STATE / (name + '.receipt.json')
    if receipt.exists():
        value = json.loads(receipt.read_text())
        if value['status'] == 'complete' and base.file_sha(path) == value['sha256']:
            return value
        raise ValueError('first download result exists; inspect without overwriting')
    if path.exists() or partial.exists():
        raise ValueError('partial or unreceipted source bytes exist')
    start = time.monotonic(); digest = hashlib.sha256(); size = 0
    try:
        with requests.get(url, stream=True, timeout=(10, 30)) as response:
            response.raise_for_status()
            declared = response.headers.get('Content-Length')
            if declared and int(declared) > MAX_FILE_BYTES:
                raise ValueError('source response declared above bounded size')
            with partial.open('xb') as handle:
                for chunk in response.iter_content(65536):
                    if time.monotonic() - start > 180:
                        raise TimeoutError('source transfer exceeded 180 seconds')
                    if not chunk:
                        continue
                    if size + len(chunk) > MAX_FILE_BYTES:
                        raise ValueError('source file byte bound exceeded')
                    handle.write(chunk); digest.update(chunk); size += len(chunk)
            if declared and size != int(declared):
                raise ValueError('complete source response length differs')
        # Read members as data only, never extract paths or execute contents.
        with tarfile.open(partial, 'r:gz') as handle:
            members = []
            for member in handle:
                if not member.isfile() or member.size > 2 * 1024 ** 3:
                    raise ValueError('unexpected source archive member')
                stream = handle.extractfile(member); decoded = hashlib.sha256(); read_bytes = 0
                for chunk in iter(lambda: stream.read(1024 ** 2), b''):
                    decoded.update(chunk); read_bytes += len(chunk)
                if read_bytes != member.size:
                    raise ValueError('source archive member is truncated')
                members.append({'name': member.name, 'bytes': member.size, 'sha256': decoded.hexdigest()})
        partial.rename(path)
        value = {'status': 'complete', 'url': url, 'local': str(path), 'bytes': size, 'sha256': digest.hexdigest(),
            'members': members, 'observed_complete_checksum_not_an_upstream_reference_match': True,
            'source_programs_executed': False, 'seconds': time.monotonic() - start}
    except Exception as exc:
        value = {'status': 'first_download_failed', 'url': url, 'bytes_received': size,
                 'error_type': type(exc).__name__, 'message': str(exc)[:500], 'partial_preserved': partial.exists(),
                 'accepted_training_rows': 0, 'seconds': time.monotonic() - start}
    base.save(receipt, value)
    print(json.dumps({k:v for k,v in value.items() if k not in ['members','url','local']}), flush=True)
    return value


def main():
    prepare()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, URLS.items()))
    if any(r['status'] != 'complete' for r in results):
        raise ValueError('original-author source fetch incomplete; first failures preserved')
    base.save(STATE / 'source-lock.json', {'files': results, 'source_plan_sha256': base.file_sha(STATE / 'source-plan.json'),
        'all_archive_members_fully_read_and_hashed': True, 'accepted_training_rows': 0,
        'needs_inventory_new_task_contract_and_first_diagnosis': True})


if __name__ == '__main__':
    main()
