"""Freeze the original public ConceptNet 5.7 assertion dump as source data."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import requests

from scripts import build_memory_sft_v2 as base

RAW = base.WORK / 'data/raw_memory_harness_v2/scale10m-conceptnet57'
STATE = base.WORK / 'reports/memory-harness-v2-conceptnet57-v1'
URL = 'https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz'
PAGES = {'downloads.html':'https://github.com/commonsense/conceptnet5/wiki/Downloads',
         'copying-and-sharing.html':'https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet'}
MAX_BYTES = 768 * 1024 ** 2


def main():
    RAW.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
    if (STATE / 'source-plan.json').exists(): raise ValueError('source fetch already started; preserve first state')
    resources = json.loads((base.REPO / 'configs/memory_sft_v2_scale_10m_resources_v2.json').read_text())['resources']
    used = sum(p.stat().st_size for p in RAW.parent.rglob('*') if p.is_file())
    if used + MAX_BYTES + 4*1024**2 > resources['new_raw_cache_cap_bytes'] or shutil.disk_usage(RAW).free < resources['minimum_free_disk_bytes'] + MAX_BYTES:
        raise ValueError('source download exceeds raw cap or disk floor')
    pages = []
    for name,url in PAGES.items():
        r=requests.get(url,timeout=(10,30));r.raise_for_status()
        if len(r.content)>2*1024**2: raise ValueError('reference page size bound')
        (RAW/name).write_bytes(r.content);pages.append({'url':url,'file':name,'sha256':base.file_sha(RAW/name)})
    head=requests.head(URL,timeout=(10,20));head.raise_for_status()
    size=int(head.headers['Content-Length'])
    if not 0<size<=MAX_BYTES: raise ValueError('source declared size outside reservation')
    base.save(STATE/'source-plan.json',{'driver_sha256':base.file_sha(Path(__file__)),
        'url':URL,'release':'5.7.0','primary_references':pages,'http_content_length':size,
        'http_etag_not_a_sha256':head.headers.get('ETag'),'max_raw_file_bytes':MAX_BYTES,
        'source_primary_declared_data_license':'cc-by-sa-4.0',
        'license_scope':'Preserve per-assertion license and sources. Complete data released under CC-BY-SA 4.0; no incompatible or noncommercial restriction may be added. Credits must be retained. Older attribution example mentions 3.0; current data-license section and per-row metadata remain authoritative evidence, not silently rewritten.',
        'published_upstream_sha256_available':False,'read_only_source_inspection':True,
        'no_source_program_execution':True,'accepted_training_rows':0,'model_requests':0,
        'created_at':datetime.now(timezone.utc).isoformat()})
    path=RAW/'conceptnet-assertions-5.7.0.csv.gz';partial=path.with_suffix(path.suffix+'.partial')
    start=time.monotonic();digest=hashlib.sha256();received=0
    with requests.get(URL,stream=True,timeout=(10,30)) as response:
        response.raise_for_status()
        with partial.open('xb') as out:
            for chunk in response.iter_content(1024**2):
                if time.monotonic()-start>180:raise TimeoutError('source download wall bound')
                received+=len(chunk)
                if received>MAX_BYTES:raise ValueError('raw source byte cap exceeded')
                out.write(chunk);digest.update(chunk)
    if received!=size:raise ValueError('source response length mismatch')
    # Complete decompression verifies the gzip trailer/CRC and records a source
    # content hash; no source rows or program text are executed.
    decoded=hashlib.sha256();uncompressed=0;lines=0
    with gzip.open(partial,'rb') as handle:
        for chunk in iter(lambda:handle.read(1024**2),b''):
            decoded.update(chunk);uncompressed+=len(chunk);lines+=chunk.count(b'\n')
            if uncompressed>32*1024**3:raise ValueError('decompressed source bound')
    partial.rename(path)
    result={'url':URL,'release':'5.7.0','local':str(path),'bytes':received,'sha256':digest.hexdigest(),
        'uncompressed_bytes':uncompressed,'uncompressed_sha256':decoded.hexdigest(),'newline_count':lines,
        'gzip_complete_crc_verified':True,'observed_sha256_not_an_upstream_reference_match':True,
        'source_plan_sha256':base.file_sha(STATE/'source-plan.json'),'seconds':time.monotonic()-start,
        'accepted_training_rows':0,'model_requests':0,'per_row_license_and_grounding_inventory_still_required':True}
    base.save(STATE/'source-lock.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
