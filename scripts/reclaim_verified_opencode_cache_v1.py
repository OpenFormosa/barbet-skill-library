"""Reclaim only reproducible raw cache after pinned-source/export verification.

All candidate text, original source locks, scores, reviews, failed attempts and
published exports remain untouched. `restore` downloads the same pinned bytes.
"""
import argparse
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi
from scripts import build_memory_sft_v2 as base
from scripts import fetch_memory_sft_v2 as fetcher
from scripts.fetch_opencode_sources_v1 import STATE as CACHE,REVISION
from scripts.publish_opencode_v26_v1 import STATE as ORIGINAL,DEST as ORIGINAL_DEST
from scripts.publish_ifm_code_v26_v1 import STATE as CURRENT

STATE=base.WORK/'reports/memory-harness-v2-cache-reclamation-v1'


def reclaim():
    STATE.mkdir(exist_ok=True)
    if (STATE/'plan.json').exists():raise ValueError('cache reclamation already started; preserve first receipts')
    lock=CACHE/'lock.json';rows=json.loads(lock.read_text())
    if len(rows)!=50 or any(r['repo']!='nvidia/OpenCodeInstruct' or r['revision']!=REVISION for r in rows):
        raise ValueError('unexpected raw cache scope')
    current=json.loads((CURRENT/'publication-final-commit.json').read_text())
    original=json.loads((ORIGINAL/'publication-final-commit.json').read_text())
    if not current['anonymous_full_file_sha256_readback'] or not original['anonymous_full_file_sha256_readback']:
        raise ValueError('complete published downstream readback required')
    source=HfApi(token=False).dataset_info('nvidia/OpenCodeInstruct',revision=REVISION,files_metadata=True)
    if source.sha!=REVISION or source.gated:raise ValueError('pinned public source not accessible')
    files={s.rfilename:s for s in source.siblings}
    for row in rows:
        p=Path(row['local']);remote=files[row['path']]
        if not p.is_relative_to(CACHE) or p.is_symlink() or not p.is_file() or p.suffix!='.parquet':
            raise ValueError('cache target outside explicit downloaded parquet scope')
        if p.stat().st_nlink!=1:raise ValueError('cache target has another hardlink; no cleanup assumed')
        if remote.lfs is None or remote.lfs.sha256!=row['sha256'] or remote.size!=row['bytes']:
            raise ValueError('pinned upstream checksum changed')
        if p.stat().st_size!=row['bytes'] or base.file_sha(p)!=row['sha256']:
            raise ValueError('raw bytes changed; preserve for diagnosis')
    manifest=json.loads((ORIGINAL/'publication-plan.json').read_text())['manifest']
    export_files=[f for f in manifest if f['path'].startswith('data/wave-0005/') or f['path'].startswith('proofs/wave-0005/')]
    if not export_files:raise ValueError('verified OpenCode exported wave not found')
    for item in export_files:
        if base.file_sha(ORIGINAL_DEST/item['path'])!=item['sha256']:raise ValueError('verified downstream export changed')
    plan={'driver_sha256':base.file_sha(Path(__file__)),'source_lock_sha256':base.file_sha(lock),
        'current_publication_sha256':base.file_sha(CURRENT/'publication-final-commit.json'),
        'original_publication_sha256':base.file_sha(ORIGINAL/'publication-final-commit.json'),
        'original_publication_final_head':original['final_head'],'current_publication_final_head':current['final_head'],
        'files':rows,'total_bytes':sum(r['bytes'] for r in rows),'verified_export_files':len(export_files),
        'all_remote_pinned_LFS_hashes_and_local_raw_hashes_match':True,'all_downstream_source_export_hashes_rechecked':True,
        'only_reconstructible_raw_cache_removed':True,'all_candidate_archives_and_first_attempt_evidence_retained':True,
        'restoration':'python -m scripts.reclaim_verified_opencode_cache_v1 restore',
        'reason':'Maintain 100 GiB free-space floor during concurrent bounded graph export and new lexical candidate construction.'}
    base.save(STATE/'plan.json',plan);before=shutil.disk_usage(base.WORK).free
    removed=[]
    for row in rows:
        Path(row['local']).unlink();removed.append(row['path'])
        base.save(STATE/'progress.json',{'removed_paths':removed,'plan_sha256':base.file_sha(STATE/'plan.json')})
    result={'removed_files':len(removed),'removed_bytes':plan['total_bytes'],'free_before':before,
        'free_after':shutil.disk_usage(base.WORK).free,'plan_sha256':base.file_sha(STATE/'plan.json'),
        'published_data_and_candidates_removed':False,'source_programs_executed':False}
    base.save(STATE/'summary.json',result);print(json.dumps(result),flush=True)


def restore():
    plan=json.loads((STATE/'plan.json').read_text());reserve=plan['total_bytes']
    if shutil.disk_usage(base.WORK).free<100*1024**3+reserve:raise ValueError('restore would breach free floor')
    if sum(p.stat().st_size for p in base.RAW.rglob('*') if p.is_file())+reserve>128*1024**3:
        raise ValueError('restore would breach raw cache cap')
    fetcher.OUT=CACHE;restored=[]
    for row in plan['files']:
        result=fetcher.fetch((row['repo'],row['revision'],row['path'],row['bytes']))
        if result['sha256']!=row['sha256']:raise ValueError('restored pinned bytes differ')
        restored.append(result)
    dest=STATE/'restored.json'
    if dest.exists():raise ValueError('restore receipt already exists')
    base.save(dest,{'plan_sha256':base.file_sha(STATE/'plan.json'),'restored_files':restored})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['reclaim','restore']);args=parser.parse_args()
    globals()[args.phase]()
