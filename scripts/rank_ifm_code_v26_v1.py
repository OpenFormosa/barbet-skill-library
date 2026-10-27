"""Explicitly bind new candidate storage to the unchanged native v26 metric."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from barbet_inference import candidate_archive as archive
from scripts import build_memory_sft_v2 as base
from scripts import normalize_ifm_code_memory_v1 as source
from scripts import prepare_v26_answer_candidates as candidates
from scripts import guard_v26_episode_identity as identity
from scripts import framed_candidate_io_v1 as framed
from scripts import freeze_v26_release_folds_v1 as freezer
from scripts.publish_ifm_math_v26_v1 import STATE as PRIOR_PUBLIC
from scripts.release_ifm_math_v26_v1 import STATE as PRIOR_SELECTED
from scripts.rank_ifm_math_v26_v1 import STATE as PRIOR_SCORE, FROZEN as ANCESTOR

STATE = source.STATE / 'v26-selection-v1'
FROZEN = source.STATE / 'frozen-prior-v1'
CALIBRATION = base.WORK / 'reports/memory-harness-v2-v26-selection'
RUNNER = base.REPO / 'scripts/score_framed_v26_candidates_v1.py'


def freeze():
    resources = json.loads(source.RESOURCES.read_text())['resources']
    reserve = 8 * 1024 ** 3
    if (source.processed_bytes() + reserve > resources['new_processed_cap_bytes']
            or shutil.disk_usage(base.WORK).free < resources['minimum_free_disk_bytes'] + reserve):
        raise ValueError('insufficient frozen-family stage storage')
    FROZEN.mkdir(exist_ok=True)
    proof = json.loads((PRIOR_PUBLIC / 'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']:
        raise ValueError('previous IFM math release not fully verified')
    package = json.loads((PRIOR_SELECTED / 'public-package-summary.json').read_text())
    proxy = FROZEN / 'prior-package-interface'; proxy.mkdir(exist_ok=True)
    base.save(proxy / 'public-package-summary.json', {**package,
        'all_published_query_folds_preserved': package['all_previous_folds_preserved'],
        'original_summary_sha256': base.file_sha(PRIOR_SELECTED / 'public-package-summary.json')})
    if not (proxy / 'public-index.sqlite').exists():
        (proxy / 'public-index.sqlite').symlink_to(PRIOR_SELECTED / 'public-index.sqlite')
    base.save(FROZEN / 'binding.json', {'verified_previous_head': proof['final_head'],
        'verified_publication_sha256': base.file_sha(PRIOR_PUBLIC / 'publication-final-commit.json'),
        'prior_public_index_sha256': base.file_sha(PRIOR_SELECTED / 'public-index.sqlite'),
        'ancestor_summary_sha256': base.file_sha(ANCESTOR / 'summary.json'),
        'freezer_code_sha256': base.file_sha(Path(freezer.__file__)), 'driver_sha256': base.file_sha(Path(__file__)),
        'all_ancestral_and_losing_answer_aliases_preserved': True, 'resource_config_sha256': base.file_sha(source.RESOURCES)})
    freezer.STATE = FROZEN; freezer.PRIOR = proxy; freezer.PRIOR_SCORE = PRIOR_SCORE; freezer.ANCESTOR = ANCESTOR
    freezer.main()


def prepare():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'candidate-input-files.json').exists():
        raise ValueError('IFM coding scoring manifest already frozen')
    summary = json.loads((source.STATE / 'normalize-summary.json').read_text())
    policy = json.loads((source.STATE / 'normalization-policy.json').read_text())
    if summary['full_normalization_policy_sha256'] != base.file_sha(source.STATE / 'normalization-policy.json'):
        raise ValueError('normalization policy differs')
    if policy['driver_sha256'] != base.file_sha(Path(source.__file__)) or policy['archive_implementation_sha256'] != base.file_sha(Path(archive.__file__)):
        raise ValueError('frozen source normalization or storage code changed')
    probe = base.WORK / 'reports/memory-harness-v2-candidate-archive-v1/index-equivalence-v1'
    comparison = json.loads((probe / 'summary.json').read_text())
    probe_plan = json.loads((probe / 'plan.json').read_text())
    if (not comparison['all_checks_passed'] or probe_plan['new_io_sha256'] != base.file_sha(Path(framed.__file__))
            or probe_plan['old_index_sha256'] != base.file_sha(Path(candidates.__file__))
            or probe_plan['old_guard_sha256'] != base.file_sha(Path(identity.__file__))):
        raise ValueError('explicit framed index equivalence not established')
    files = []; rows = 0
    for job in json.loads((source.STATE / 'normalization-jobs.json').read_text()):
        report = json.loads((source.STATE / 'normalized' / ('full-' + job['job_id'] + '.report.json')).read_text())
        if (report['policy_sha256'] != summary['full_normalization_policy_sha256']
                or base.file_sha(report['path']) != report['sha256'] or not report['all_frames_exactly_reread']):
            raise ValueError('complete candidate archive readback missing')
        count = report['counts'].get('candidate_answers', 0); rows += count
        if count:
            files.append({'file_id': len(files), 'path': report['path'], 'sha256': report['sha256'],
                          'format': archive.FORMAT, 'wave': 'ifm-coding-context-v1'})
    if rows != summary['counts']['candidate_answers']:
        raise ValueError('complete eligible answer version coverage differs')
    first = json.loads((source.SOURCE / 'semantic-audit-v1/summary.json').read_text())
    base.save(STATE / 'candidate-holds.private.json', {'record_ids': first['excluded_record_ids'],
        'query_hashes': policy['first_nonaccept_query_hashes'], 'search_allowed': [],
        'supplemental_prior_queries': policy['unequal_literal_query_aliases']})
    for name in ['canonical-single-v3-validation.json', 'live-calibration-diagnostic-v3.json', 'remote-v26-tag-check.json']:
        shutil.copyfile(CALIBRATION / name, STATE / name)
    base.save(STATE / 'candidate-input-files.json', files)
    bindings = {'driver_sha256': base.file_sha(Path(__file__)),
        'policy_sha256': base.file_sha(candidates.POLICY), 'normalization_summary_sha256': base.file_sha(source.STATE / 'normalize-summary.json'),
        'first_source_audit_sha256': base.file_sha(source.SOURCE / 'semantic-audit-v1/summary.json'),
        'canonicalizer_sha256': base.file_sha(Path(candidates.__file__)),
        'structural_guard_algorithm_sha256': base.file_sha(Path(identity.__file__)),
        'storage_loader_sha256': base.file_sha(Path(archive.__file__)), 'storage_format': archive.FORMAT,
        'framed_index_adapter_sha256': base.file_sha(Path(framed.__file__)),
        'native_runner_sha256': base.file_sha(RUNNER),
        'unchanged_numerical_scorer_sha256': base.file_sha(base.REPO / 'barbet_inference/answer_selection.py'),
        'framed_index_equivalence_proof_sha256': base.file_sha(probe / 'summary.json'),
        'structural_identity_regression_log_sha256': base.file_sha(base.STATE / 'framed-candidate-identity-tests-v1.log'),
        'all_first_holds_preserved': True, 'native_v26_minimum_conditional_answer_ppl': True,
        'all_eligible_source_answer_versions_in_manifest': True, 'source_programs_executed': False, 'weights_trained': False}
    base.save(STATE / 'candidate-policy-binding.json', bindings)
    print(json.dumps({'files': len(files), 'candidate_answers': rows, 'storage_format': archive.FORMAT}), flush=True)


def configure():
    binding = json.loads((STATE / 'candidate-policy-binding.json').read_text())
    for key, path in [('driver_sha256', Path(__file__)), ('canonicalizer_sha256', Path(candidates.__file__)),
                      ('structural_guard_algorithm_sha256', Path(identity.__file__)),
                      ('storage_loader_sha256', Path(archive.__file__)), ('framed_index_adapter_sha256', Path(framed.__file__)),
                      ('native_runner_sha256', RUNNER)]:
        if binding[key] != base.file_sha(path):
            raise ValueError('prospective framed cohort code binding differs: ' + key)
    # Only new cohort byte readers are injected; all grouping/identity SQL and
    # numerical functions remain the originals bound above.
    candidates.STATE = STATE; candidates.index_file = framed.index_file
    identity.STATE = STATE / 'episode-identity-v1'; identity.job = framed.guard_file


def score_all():
    proof = json.loads((STATE / 'episode-identity-v1/summary.json').read_text())
    if proof['status'] != 'COMPLETE' or not proof['all_original_candidates_covered']:
        raise ValueError('complete episode identity coverage required')
    folder = STATE / 'scores/canonical-v2'; folder.mkdir(parents=True, exist_ok=True)
    receipt = STATE / 'score-launch-v1.json'
    if receipt.exists():
        raise ValueError('native score launch already recorded; inspect without replay')
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    handles = []; children = []
    for bucket in range(8):
        handle = (folder / f'bucket-{bucket}.log').open('w'); handles.append(handle)
        command = ['/root/Barbet/.venv/bin/python', str(RUNNER), '--state', str(STATE), '--bucket', str(bucket)]
        process = subprocess.Popen(command, cwd=base.REPO, env={**env, 'CUDA_VISIBLE_DEVICES': str(bucket)}, stdout=handle, stderr=subprocess.STDOUT)
        children.append((bucket, process))
    base.save(receipt, {'driver_sha256': base.file_sha(Path(__file__)), 'native_runner_sha256': base.file_sha(RUNNER),
        'pids': [{'bucket': b, 'pid': p.pid} for b, p in children], 'canonical_unpadded_batch_size': 1,
        'max_gpu_workers': 8, 'runtime_dependencies_modified': False, 'weights_trained': False})
    failures = []
    for bucket, process in children:
        code = process.wait(); print(json.dumps({'bucket': bucket, 'exit_code': code}), flush=True)
        if code:
            failures.append(bucket)
    for handle in handles:
        handle.close()
    if failures:
        raise RuntimeError('native scoring incomplete: ' + str(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'prepare', 'index', 'guard', 'score-all'])
    parser.add_argument('--workers', type=int, default=4); args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('candidate indexing worker bound')
    if args.phase == 'freeze':
        freeze()
    elif args.phase == 'prepare':
        prepare()
    else:
        configure()
        if args.phase == 'index':
            if (STATE / 'candidate-index-summary.json').exists():
                raise ValueError('candidate index already frozen')
            candidates.index(args.workers)
        elif args.phase == 'guard':
            identity.main()
        else:
            score_all()
