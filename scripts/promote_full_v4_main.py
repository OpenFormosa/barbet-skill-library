"""Publish the complete report card, then fast-forward HF main without force."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import ast
import json
import os
import re
import subprocess
import yaml
from huggingface_hub import HfApi, CommitOperationAdd
from barbet_pipeline.common import read_json, save_new, sha_file
from barbet_pipeline.anonymous_readback_ranged import verify_remote

CODE = Path('/root/barbet-full-corpus-v4')
STATE = Path('/root/barbet-context-sft-data/reports/full-corpus-v4')
WORK = STATE / 'main-promotion-20261105'
REPO = 'OpenFormosa/barbet-long-context-sft'
URL = 'https://huggingface.co/datasets/' + REPO
V4 = '7103d1b1bc1fb4cda1298c084a66bda542d61151'
OLD = 'b88f4ff7171bb6fb8c1b775ae21de814ea2123fb'
BACKUP = 'archive/main-before-v4-20261105'
CANDIDATE = 'v4-main-readme-20261105'
PROMOTION_PATH = 'full-v4/main-promotion-20261105.json'
PYTHON = '/root/barbet-context-sft-data/.venv/bin/python'


def checked_git(*arguments, env=None):
    return subprocess.run(['git', '-c', 'http.version=HTTP/1.1', '-c', 'credential.helper=',
        '-C', str(WORK / 'hf.git'), *arguments], check=True, text=True, capture_output=True,
        env=env, timeout=180)


def check_readme():
    text = (WORK / 'README.md').read_text()
    front = yaml.safe_load(text.split('---', 2)[1])
    frozen = yaml.safe_load((STATE / 'release-package/README.md').read_text().split('---', 2)[1])
    if front['configs'] != frozen['configs']:
        raise ValueError('The promotion must preserve all v4 configurations and splits')
    if text.count(chr(96)*3) % 2:
        raise ValueError('Unclosed Markdown code fence')
    for snippet in re.findall(chr(96)*3 + r'python\n(.*?)\n' + chr(96)*3, text, re.S):
        ast.parse(snippet)
    for snippet in re.findall(chr(96)*3 + r'json\n(.*?)\n' + chr(96)*3, text, re.S):
        json.loads(snippet)
    for required in ['10,018,967', '10,018,961', '30,174,631,335', '3,951,574,820',
            '89.21%', '97.05%', '1.093813', '0/287', '0/286', 'SIGTERM',
            '76.25%', 'training_ready=false', 'source_reference', 'reference_only', V4,
            '## 十三、文獻依據', '## 六、資料格式', '## 十、如何載入']:
        if required not in text:
            raise ValueError('Missing factual report content: ' + required)
    local_links = {link.split('#', 1)[0] for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text)
        if not urlparse(link).scheme and not link.startswith('#')}
    return text, front, local_links


def smoke(revision, output, log):
    if output.exists():
        saved = read_json(output)
        if saved['revision'] != revision or not saved['all_configs_and_splits_loaded']:
            raise ValueError('Stale or incomplete configuration controls')
        return
    with log.open('x') as handle:
        subprocess.run([PYTHON, str(CODE / 'scripts/smoke_full_v4_hub.py'), revision, str(output)],
            cwd=CODE, env=dict(os.environ, PYTHONPATH=str(CODE)), stdout=handle,
            stderr=subprocess.STDOUT, check=True)


def main():
    api = HfApi()
    text, front, links = check_readme()
    if api.repo_info(REPO, repo_type='dataset', revision='full-corpus-v4-20261104').sha != V4:
        raise ValueError('The frozen v4 source branch moved')
    info = api.get_paths_info(REPO, sorted(links - {PROMOTION_PATH}), repo_type='dataset', revision=V4)
    if {x.path for x in info} != links - {PROMOTION_PATH}:
        raise ValueError('README references a missing v4 evidence file')
    readme = WORK / 'README.md'
    descriptor = WORK / 'main-promotion-20261105.json'
    if not descriptor.exists():
        save_new(descriptor, {'role': 'Immutable data/readme identity contract for the commit promoted to main.',
            'prepared_at_utc': datetime.now(timezone.utc).isoformat(), 'source_v4_revision': V4,
            'previous_main_revision': OLD, 'previous_main_archive_branch': BACKUP,
            'candidate_branch': CANDIDATE, 'target_branch': 'main',
            'allowed_changed_paths_from_v4': ['README.md', PROMOTION_PATH],
            'readme_sha256': sha_file(readme), 'readme_bytes': readme.stat().st_size,
            'source_manifest_sha256': sha_file(STATE / 'release-package/release-manifest.json'),
            'source_publication_receipt_sha256': sha_file(STATE / 'release-package/publication-complete.json'),
            'original_manifest_readme_hash_scope': 'The original source_v4_revision; this documentation successor does not rewrite that historical receipt.',
            'data_payload_changes': 0, 'gpu_measurement_payload_changes': 0,
            'source_rows': 10018967, 'exported_rows': 10018961, 'quarantined_rows': 6,
            'gpu_measured_rows': 10018961, 'training_ready': False,
            'main_update_method': 'Git fast-forward only, without force or history rewrite.',
            'renderer_sha256': sha_file(CODE / 'scripts/write_full_v4_main_readme.py'),
            'promotion_implementation_sha256': sha_file(__file__)})
    saved = read_json(descriptor)
    if saved['readme_sha256'] != sha_file(readme):
        raise ValueError('The frozen promotion README changed')
    branches = {x.name: x.target_commit for x in api.list_repo_refs(REPO, repo_type='dataset').branches}
    if BACKUP in branches and branches[BACKUP] != OLD:
        raise ValueError('Preserve the existing backup ref; its target differs')
    if BACKUP not in branches:
        api.create_branch(REPO, repo_type='dataset', branch=BACKUP, revision=OLD)
    staged = WORK / 'candidate-commit.json'
    if not staged.exists():
        if CANDIDATE in branches:
            raise ValueError('An unrecorded candidate ref already exists; preserve it')
        api.create_branch(REPO, repo_type='dataset', branch=CANDIDATE, revision=V4)
        result = api.create_commit(REPO, repo_type='dataset', revision=CANDIDATE, parent_commit=V4,
            operations=[CommitOperationAdd('README.md', str(readme)),
                        CommitOperationAdd(PROMOTION_PATH, str(descriptor))],
            commit_message='Make the complete v4 data and GPU report the main dataset card')
        save_new(staged, {'head': result.oid, 'parent': V4, 'readme_sha256': sha_file(readme)})
    head = read_json(staged)['head']
    if api.repo_info(REPO, repo_type='dataset', revision=CANDIDATE).sha != head:
        raise ValueError('The candidate branch moved')
    checked_git('fetch', '--filter=blob:none', '--no-tags', URL,
        head + ':refs/remotes/source/readme')
    checked_git('merge-base', '--is-ancestor', OLD, head)
    changed = checked_git('diff', '--name-only', V4, head).stdout.splitlines()
    if set(changed) != {'README.md', PROMOTION_PATH}:
        raise ValueError('A v4 payload changed during documentation preparation')
    staged_readback = WORK / 'candidate-anonymous-readback.json'
    if not staged_readback.exists():
        items = [{'local': str(path), 'remote': remote, 'sha256': sha_file(path), 'bytes': path.stat().st_size}
            for path, remote in [(readme, 'README.md'), (descriptor, PROMOTION_PATH)]]
        save_new(staged_readback, {'revision': head, 'files': [verify_remote(i, head) for i in items],
            'exact_changed_paths': changed})
    print(json.dumps({'status': 'candidate_report_byte_verified', 'revision': head,
        'readme_bytes': readme.stat().st_size}), flush=True)
    smoke(head, WORK / 'candidate-config-loads.json', WORK / 'candidate-config-loads.log')
    current_main = api.repo_info(REPO, repo_type='dataset', revision='main').sha
    if current_main not in {OLD, head}:
        raise ValueError('main moved independently; do not overwrite it')
    if current_main == OLD:
        askpass = WORK / 'git-askpass.py'
        if not askpass.exists():
            askpass.write_text('#!' + PYTHON + '\nimport os,sys\nfrom huggingface_hub import get_token\n'
                'prompt=sys.argv[1].lower()\n'
                'value=os.environ["HF_PROMOTION_GIT_USER"] if "username" in prompt else get_token()\n'
                'if not value:sys.exit(1)\n'
                'sys.stdout.write(value+"\\n")\n')
            askpass.chmod(0o700)
        env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_ASKPASS=str(askpass),
            HF_PROMOTION_GIT_USER=api.whoami()['name'], GIT_TRACE_REDACT='1')
        for name in ['GIT_TRACE', 'GIT_CURL_VERBOSE', 'GIT_TRACE_CURL', 'GIT_TRACE_PACKET']:
            env.pop(name, None)
        result = checked_git('push', URL, head + ':refs/heads/main', env=env)
        save_new(WORK / 'main-fast-forward.json', {'from': OLD, 'to': head,
            'stdout': result.stdout, 'stderr': result.stderr,
            'method': 'Non-force Git fast-forward', 'candidate_controls_sha256': sha_file(WORK / 'candidate-config-loads.json')})
    if api.repo_info(REPO, repo_type='dataset', revision='main').sha != head:
        raise ValueError('main is not the fully verified report commit')
    if api.repo_info(REPO, repo_type='dataset', revision=BACKUP).sha != OLD:
        raise ValueError('The old main backup moved')
    if api.file_exists(REPO, 'FULL_V4_STAGING_INCOMPLETE.txt', repo_type='dataset', revision='main'):
        raise ValueError('An incomplete marker is present on main')
    smoke('main', WORK / 'main-config-loads.json', WORK / 'main-config-loads.log')
    checked = verify_remote({'remote': 'README.md', 'local': str(readme),
        'sha256': sha_file(readme), 'bytes': readme.stat().st_size}, head)
    result = {'status': 'V4_MAIN_AND_COMPLETE_README_PUBLISHED',
        'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'main_revision': head,
        'v4_source_revision': V4, 'old_main_revision': OLD, 'old_main_backup_branch': BACKUP,
        'readme_sha256': sha_file(readme), 'readme_bytes': readme.stat().st_size,
        'data_and_measurement_paths_changed': 0, 'exact_changed_paths': changed,
        'candidate_config_split_controls': len(read_json(WORK / 'candidate-config-loads.json')['controls']),
        'main_config_split_controls': len(read_json(WORK / 'main-config-loads.json')['controls']),
        'main_readme_anonymous_readback': checked, 'source_v4_branch_preserved': True}
    save_new(WORK / 'completion.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
