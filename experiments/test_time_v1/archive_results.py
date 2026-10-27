#!/usr/bin/env python3
"""Package the committed source and local evidence; never upload benchmark data."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8*1024*1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    result = json.loads((HERE / 'results.json').read_text())
    if result['status'] != 'completed':
        raise RuntimeError('archive only after all sealed results and analyses are complete')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('commit reviewed reports before packaging the source snapshot')
    work = HERE / 'work'
    files = set()
    for folder in ['runs', 'call_logs', 'scores', 'logs', 'context_probe']:
        files.update(p for p in (work / folder).rglob('*') if p.is_file())
    for pattern in ['*.jsonl', '*.json']:
        files.update(p for p in work.glob(pattern) if p.is_file())
    source = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    source = [ROOT / s for s in source if s]
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = {'source_git_sha': git_sha, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'notice': 'Local reproducibility bundle. Original dataset terms remain separate; not uploaded to the public skill repository. Model weights are referenced by pinned revisions and hashes, not bundled.',
        'files': [{'path': 'evidence/' + str(p.relative_to(work)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(files)]}
    target_dir = work / 'artifacts'
    target_dir.mkdir(exist_ok=True)
    manifest_path = target_dir / 'raw_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    archive = target_dir / 'barbet-test-time-v1.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=6, dereference=True) as tar:
        for path in source:
            tar.add(path, arcname='repository/' + str(path.relative_to(ROOT)), recursive=False)
        for path in sorted(files):
            tar.add(path, arcname='evidence/' + str(path.relative_to(work)), recursive=False)
        tar.add(manifest_path, arcname='raw_manifest.json', recursive=False)
    receipt = {'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive),
        'source_git_sha': git_sha, 'evidence_files': len(files),
        'raw_manifest_sha256': digest(manifest_path), 'published_to_github': False}
    (HERE / 'archive_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
