"""Bounded immutable source download; importing a source never accepts its rows."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import requests
from scripts.build_memory_sft_v2 import RAW, WORK, save, file_sha


def fetch(entry):
    dest=Path(entry['local']);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if dest.stat().st_size!=entry['bytes']:raise ValueError('existing source size mismatch')
        return {**entry,'sha256':file_sha(dest)}
    url=f'https://huggingface.co/datasets/{entry["repo"]}/resolve/{entry["revision"]}/{entry["path"]}'
    part=dest.with_suffix(dest.suffix+'.partial')
    h=hashlib.sha256();size=0
    with requests.get(url,stream=True,timeout=(30,180)) as response:
        response.raise_for_status()
        with part.open('wb') as f:
            for chunk in response.iter_content(8*1024**2):
                size+=len(chunk)
                if size>entry['bytes']:raise ValueError('unexpected upstream size')
                h.update(chunk);f.write(chunk)
    if size!=entry['bytes']:raise ValueError('incomplete source download')
    part.replace(dest)
    return {**entry,'sha256':h.hexdigest()}


def main():
    sid='IFM/TxT360-3efforts';revision='bfc4a082d11967cd7810fe0b773be87bf54fb32e'
    api=json.loads((RAW/'scale10m-reference/IFM--TxT360-3efforts.api.json').read_text())
    if api['sha']!=revision or api.get('gated') or api.get('cardData',{}).get('license')!='cc-by-4.0':
        raise ValueError('source contract changed')
    selected={'agent','code','math','math-dialogue','teacher-as-assistant','instructions-with-constraints','stem','chat'}
    root=RAW/'scale10m-ifm';plan=[]
    for f in api['siblings']:
        path=f['rfilename']
        if path.endswith('.parquet') and path.split('/')[0] in selected:
            plan.append({'repo':sid,'revision':revision,'path':path,'bytes':f['size'],'local':str(root/path)})
    # Conservative logical-byte accounting includes hard-linked source copies.
    existing=sum(p.stat().st_size for p in RAW.rglob('*') if p.is_file())
    additional=sum(f['bytes'] for f in plan if not Path(f['local']).exists())
    if existing+additional>128*1024**3:raise ValueError('128 GiB new raw cache limit')
    if shutil.disk_usage(WORK).free-additional<100*1024**3:raise ValueError('100 GiB free disk reserve')
    save(root/'plan.json',{'source':sid,'revision':revision,'selected_prefixes':sorted(selected),
        'files':len(plan),'bytes':sum(f['bytes'] for f in plan),'raw_existing_bytes':existing,
        'no_training_rows_accepted_by_download':True,'upstream_turn_views_and_effort_variants_must_not_inflate_count':True})
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,result in enumerate(pool.map(fetch,plan),1):
            results.append(result)
            if n%10==0 or n==len(plan):print(json.dumps({'downloaded':n,'total':len(plan)}),flush=True)
    save(root/'lock.json',results)


if __name__=='__main__':main()
