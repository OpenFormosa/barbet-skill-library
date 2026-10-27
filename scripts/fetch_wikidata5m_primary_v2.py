"""Use recorded primary web-policy evidence after the raw policy-page 403."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from scripts import build_memory_sft_v2 as base
from scripts import fetch_wikidata5m_primary_v1 as transport


def prepare():
    state = transport.STATE; raw = transport.RAW
    path = state / 'source-plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['driver_sha256'] != base.file_sha(Path(__file__)) or plan['download_transport_sha256'] != base.file_sha(Path(transport.__file__)):
            raise ValueError('prospective source download binding differs')
        return
    failure_path = state / 'first-reference-http-failure-v1.json'
    failure = json.loads(failure_path.read_text())
    if failure['archive_requests_started'] or failure['source_plan_created'] or failure['driver_sha256'] != base.file_sha(Path(transport.__file__)):
        raise ValueError('first failure boundary differs')
    license_path = state / 'official-license-web-evidence-v1.json'
    licensing = json.loads(license_path.read_text())
    if licensing['structured_data_license'] != 'CC0-1.0' or licensing['primary_url'] != transport.PAGES['wikidata-structured-license.html']:
        raise ValueError('primary structured-data policy evidence missing')
    resources = json.loads((base.REPO / 'configs/memory_sft_v2_scale_10m_resources_v2.json').read_text())['resources']
    used = sum(p.stat().st_size for p in raw.parent.rglob('*') if p.is_file()); reserve = 2 * transport.MAX_FILE_BYTES
    if used + reserve > resources['new_raw_cache_cap_bytes'] or shutil.disk_usage(raw).free < resources['minimum_free_disk_bytes'] + reserve:
        raise ValueError('source download resource bound')
    base.save(path, {'driver_sha256': base.file_sha(Path(__file__)), 'download_transport_sha256': base.file_sha(Path(transport.__file__)),
        'first_reference_failure_sha256': base.file_sha(failure_path), 'official_license_web_evidence_sha256': base.file_sha(license_path),
        'urls': transport.URLS, 'original_project_page_sha256': base.file_sha(raw / 'original-project.html'),
        'original_project_url': transport.PAGES['original-project.html'], 'raw_bytes_before_download': used,
        'reserved_download_bytes': reserve, 'maximum_file_bytes': transport.MAX_FILE_BYTES,
        'original_source_expected_inductive_train_entities': 4579609,
        'original_source_expected_inductive_train_triples': 20496514,
        'published_upstream_checksum_available': False,
        'license_scope': 'Official Wikidata CC0 structured triples and aliases only; Wikipedia prose is outside this source investigation.',
        'split_scope': 'Inductive training graph only for future evidence. Validation/test are retained only for source inventory and exclusion checks.',
        'accepted_training_rows': 0, 'source_programs_executed': False, 'model_requests': 0,
        'created_at': datetime.now(timezone.utc).isoformat()})


def main():
    prepare()
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(transport.fetch, transport.URLS.items()))
    if any(r['status'] != 'complete' for r in result):
        raise ValueError('original source archives incomplete; preserve first results')
    base.save(transport.STATE / 'source-lock.json', {'files': result,
        'source_plan_sha256': base.file_sha(transport.STATE / 'source-plan.json'),
        'all_archive_members_fully_read_and_hashed': True, 'accepted_training_rows': 0,
        'needs_inventory_new_task_contract_and_first_diagnosis': True})


if __name__ == '__main__':
    main()
