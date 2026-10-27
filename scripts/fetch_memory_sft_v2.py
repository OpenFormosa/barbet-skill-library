"""Fetch pinned inert source bytes. Never imports or executes downloaded programs."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path('/root/barbet-context-sft-data')
OUT = ROOT / 'data/raw_memory_harness_v2'
SOURCES = {
    'voidful/agent-sft': ('8f7a97fcb0fa3cba16139fd1925e02febd4980ad', lambda p: p.endswith('.parquet')),
    'nvidia/Nemotron-SFT-Agentic-v2': ('7c804833427f633ccd53b582dbf02525fd680f78', lambda p: p.endswith('.jsonl')),
    'hotpotqa/hotpot_qa': ('1908d6afbbead072334abe2965f91bd2709910ab', lambda p: p.startswith('distractor/train-') and p.endswith('.parquet')),
    'rajpurkar/squad_v2': ('3ffb306f725f7d2ce8394bc1873b24868140c412', lambda p: p.startswith('squad_v2/train-') and p.endswith('.parquet')),
}

def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()

def fetch(job):
    repo, rev, remote, expected = job
    target = OUT / repo.replace('/', '--') / remote
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        tmp = target.with_suffix(target.suffix + '.partial')
        url = f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{remote}'
        with urllib.request.urlopen(url, timeout=300) as src, tmp.open('wb') as dest:
            shutil.copyfileobj(src, dest, 8 * 1024 * 1024)
        if expected is not None and tmp.stat().st_size != expected:
            raise ValueError('SOURCE_SIZE_MISMATCH:' + remote)
        tmp.rename(target)
    sha = hashlib.sha256()
    with target.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            sha.update(block)
    result = {'repo': repo, 'revision': rev, 'path': remote, 'local': str(target),
              'bytes': target.stat().st_size, 'sha256': sha.hexdigest()}
    print(json.dumps(result), flush=True)
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for repo, (rev, include) in SOURCES.items():
        raw = get(f'https://huggingface.co/api/datasets/{repo}/revision/{rev}?blobs=true')
        api = json.loads(raw)
        if api['sha'] != rev:
            raise ValueError('SOURCE_REVISION_MISMATCH')
        folder = OUT / repo.replace('/', '--')
        folder.mkdir(exist_ok=True)
        (folder / 'api.json').write_bytes(raw)
        for f in api['siblings']:
            name = f['rfilename']
            if include(name) or name in ('README.md', 'LICENSE', 'LICENSE.txt', '_report.json'):
                jobs.append((repo, rev, name, f.get('size')))
    expected = sum(j[3] or 0 for j in jobs)
    if expected > 64 * 1024**3 or shutil.disk_usage(OUT).free < 3 * expected:
        raise ValueError('RESOURCE_BUDGET_EXCEEDED')
    print(json.dumps({'prospective_download_bytes': expected, 'files': len(jobs)}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, jobs))
    (OUT / 'sources.lock.json').write_text(json.dumps({'files': results}, indent=2) + '\n')

if __name__ == '__main__':
    main()
