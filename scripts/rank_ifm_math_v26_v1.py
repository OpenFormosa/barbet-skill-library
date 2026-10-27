"""Complete native-v26 math answer ranking; no new scorer or numerical recipe."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts import build_memory_sft_v2 as base
from scripts import prepare_v26_answer_candidates as candidates
from scripts import guard_v26_episode_identity as identity
from scripts import freeze_v26_release_folds_v1 as freezer
from scripts import normalize_ifm_math_memory_v1 as source
from scripts.freeze_opencode_prior_v1 import STATE as ANCESTOR
from scripts.finalize_opencode_v26_v1 import STATE as OPENCODE
from scripts.rank_opencode_v26_v1 import STATE as OPENCODE_SCORE
from scripts.publish_opencode_v26_v1 import STATE as OPENCODE_PUBLIC

STATE = source.STATE / 'v26-selection-v1'
FROZEN = source.STATE / 'frozen-prior-v1'
CALIBRATION = base.WORK / 'reports/memory-harness-v2-v26-selection'


def freeze():
    FROZEN.mkdir(exist_ok=True)
    proof = json.loads((OPENCODE_PUBLIC / 'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']: raise ValueError('OpenCode release not verified')
    package = json.loads((OPENCODE / 'public-package-summary.json').read_text())
    proxy = FROZEN / 'prior-package-interface'; proxy.mkdir(exist_ok=True)
    base.save(proxy / 'public-package-summary.json', {**package,
        'all_published_query_folds_preserved': package['all_previous_folds_preserved'],
        'original_summary_sha256': base.file_sha(OPENCODE / 'public-package-summary.json')})
    if not (proxy / 'public-index.sqlite').exists(): (proxy / 'public-index.sqlite').symlink_to(OPENCODE / 'public-index.sqlite')
    base.save(FROZEN / 'binding.json', {'verified_opencode_head': proof['final_head'],
        'verified_publication_sha256': base.file_sha(OPENCODE_PUBLIC / 'publication-final-commit.json'),
        'opencode_public_index_sha256': base.file_sha(OPENCODE / 'public-index.sqlite'),
        'ancestor_summary_sha256': base.file_sha(ANCESTOR / 'summary.json'),
        'freezer_code_sha256': base.file_sha(Path(freezer.__file__)), 'driver_sha256': base.file_sha(Path(__file__)),
        'all_ancestral_and_losing_answer_aliases_preserved': True,
        'removed_source_contract_queries_remain_conservatively_frozen': True})
    freezer.STATE = FROZEN; freezer.PRIOR = proxy; freezer.PRIOR_SCORE = OPENCODE_SCORE; freezer.ANCESTOR = ANCESTOR
    freezer.main()


def prepare():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'candidate-input-files.json').exists(): raise ValueError('IFM math scoring manifest already frozen')
    summary = json.loads((source.STATE / 'normalize-summary.json').read_text())
    policy = json.loads((source.STATE / 'normalization-policy.json').read_text())
    if summary['full_normalization_policy_sha256'] != base.file_sha(source.STATE / 'normalization-policy.json'):
        raise ValueError('normalization policy changed')
    if policy['driver_sha256'] != base.file_sha(Path(source.__file__)):
        raise ValueError('frozen full normalizer changed')
    if policy['adapter_sha256'] != base.file_sha(base.REPO / 'scripts/ifm_memory_adapter_v1.py'):
        raise ValueError('frozen first-audited adapter changed')
    files = []; rows = 0
    for spec in json.loads((source.STATE / 'normalization-jobs.json').read_text()):
        report = json.loads((source.STATE / 'normalized' / ('full-' + spec['job_id'] + '.report.json')).read_text())
        if report['policy_sha256'] != summary['full_normalization_policy_sha256'] or base.file_sha(report['path']) != report['sha256']:
            raise ValueError('candidate shard binding differs')
        rows += report['counts'].get('candidate_answers', 0)
        if report['counts'].get('candidate_answers', 0):
            files.append({'file_id': len(files), 'path': report['path'], 'sha256': report['sha256'], 'wave': 'ifm-math-context-v1'})
    if rows != summary['counts']['candidate_answers']: raise ValueError('candidate coverage differs')
    audit = json.loads((source.SOURCE / 'semantic-audit-v1/summary.json').read_text())
    base.save(STATE / 'candidate-holds.private.json', {'record_ids': audit['excluded_record_ids'],
        'query_hashes': policy['first_nonaccept_query_hashes'], 'search_allowed': [],
        'supplemental_prior_queries': policy['unequal_literal_query_aliases']})
    for name in ['canonical-single-v3-validation.json', 'live-calibration-diagnostic-v3.json', 'remote-v26-tag-check.json']:
        shutil.copyfile(CALIBRATION / name, STATE / name)
    base.save(STATE / 'candidate-input-files.json', files)
    base.save(STATE / 'candidate-policy-binding.json', {'driver_sha256': base.file_sha(Path(__file__)),
        'policy_sha256': base.file_sha(candidates.POLICY), 'normalization_summary_sha256': base.file_sha(source.STATE / 'normalize-summary.json'),
        'first_source_audit_sha256': base.file_sha(source.SOURCE / 'semantic-audit-v1/summary.json'),
        'structural_guard_driver_sha256': base.file_sha(Path(identity.__file__)),
        'all_first_holds_preserved': True, 'native_v26_minimum_conditional_answer_ppl': True,
        'all_eligible_source_effort_versions_in_manifest': True, 'source_programs_executed': False, 'weights_trained': False})
    print(json.dumps({'files': len(files), 'candidate_answers': rows}), flush=True)


def configure():
    binding = json.loads((STATE / 'candidate-policy-binding.json').read_text())
    if binding['driver_sha256'] != base.file_sha(Path(__file__)): raise ValueError('frozen IFM scoring driver changed')
    if binding['structural_guard_driver_sha256'] != base.file_sha(Path(identity.__file__)):
        raise ValueError('structural guard changed')
    candidates.STATE = STATE; identity.STATE = STATE / 'episode-identity-v1'


def score_all():
    if json.loads((STATE / 'episode-identity-v1/summary.json').read_text())['status'] != 'COMPLETE':
        raise ValueError('complete structural identity coverage required')
    folder = STATE / 'scores/canonical-v2'; folder.mkdir(parents=True, exist_ok=True)
    if (STATE / 'score-launch-v1.json').exists(): raise ValueError('scoring launch already recorded')
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    children = []; handles = []
    for bucket in range(8):
        handle = (folder / f'bucket-{bucket}.log').open('w'); handles.append(handle)
        command = ['/root/Barbet/.venv/bin/python', str(Path(__file__).resolve()), 'score', '--bucket', str(bucket)]
        proc = subprocess.Popen(command, cwd=base.REPO, env={**env, 'CUDA_VISIBLE_DEVICES': str(bucket)}, stdout=handle, stderr=subprocess.STDOUT)
        children.append((bucket, proc))
    base.save(STATE / 'score-launch-v1.json', {'pids': [{'bucket': b, 'pid': p.pid} for b, p in children],
        'native_unpadded_batch_size': 1, 'max_gpu_workers': 8, 'weights_trained': False})
    failed = []
    for bucket, process in children:
        code = process.wait(); print(json.dumps({'bucket': bucket, 'exit_code': code}), flush=True)
        if code: failed.append(bucket)
    for handle in handles: handle.close()
    if failed: raise RuntimeError('incomplete scoring buckets: ' + str(failed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'prepare', 'index', 'guard', 'score-all', 'score'])
    parser.add_argument('--workers', type=int, default=4); parser.add_argument('--bucket', type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: raise ValueError('index worker bound')
    if args.phase == 'freeze': freeze()
    elif args.phase == 'prepare': prepare()
    else:
        configure()
        if args.phase == 'index':
            if (STATE / 'candidate-index-summary.json').exists(): raise ValueError('index already frozen')
            candidates.index(args.workers)
        elif args.phase == 'guard': identity.main()
        elif args.phase == 'score-all': score_all()
        else:
            if args.bucket is None: raise ValueError('score bucket required')
            from scripts import score_v26_answer_candidates as scorer
            scorer.STATE = STATE; scorer.main(args.bucket, 0, 'canonical-v2')
