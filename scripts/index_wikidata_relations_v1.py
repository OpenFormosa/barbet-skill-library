"""Bind complete relation candidates to the existing identity and fold guards."""
import argparse
import json
from pathlib import Path
import shutil

from barbet_inference import candidate_archive as archive
from scripts import build_memory_sft_v2 as base
from scripts import normalize_wikidata_relations_v1 as source
from scripts import prepare_v26_answer_candidates as candidates
from scripts import guard_v26_episode_identity as identity
from scripts import framed_candidate_io_v1 as framed
from scripts import freeze_v26_release_folds_v1 as freezer
from scripts.rank_ifm_code_v26_v1 import FROZEN as ANCESTOR, STATE as PRIOR_SCORE
from scripts.release_ifm_code_v26_v1 import STATE as PRIOR_SELECTED
from scripts.publish_ifm_code_v26_v1 import STATE as PRIOR_PUBLIC

STATE=source.STATE/'candidate-selection-v1'
FROZEN=source.STATE/'frozen-prior-v1'


def freeze():
    resources=json.loads(source.RESOURCES.read_text())['resources'];reserve=8*1024**3
    if source.processed_bytes()+reserve>resources['new_processed_cap_bytes'] or shutil.disk_usage(base.WORK).free<resources['minimum_free_disk_bytes']+reserve:
        raise ValueError('frozen prior identity stage resource bound')
    proof=json.loads((PRIOR_PUBLIC/'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']:raise ValueError('prior published release unverified')
    FROZEN.mkdir(exist_ok=True);proxy=FROZEN/'prior-package-interface';proxy.mkdir(exist_ok=True)
    package=json.loads((PRIOR_SELECTED/'public-package-summary.json').read_text())
    base.save(proxy/'public-package-summary.json',{**package,
        'all_published_query_folds_preserved':package['all_previous_folds_preserved'],
        'original_summary_sha256':base.file_sha(PRIOR_SELECTED/'public-package-summary.json')})
    if not (proxy/'public-index.sqlite').exists():(proxy/'public-index.sqlite').symlink_to(PRIOR_SELECTED/'public-index.sqlite')
    base.save(FROZEN/'binding.json',{'driver_sha256':base.file_sha(Path(__file__)),
        'freezer_sha256':base.file_sha(Path(freezer.__file__)),
        'prior_verified_publication_sha256':base.file_sha(PRIOR_PUBLIC/'publication-final-commit.json'),
        'prior_selected_index_sha256':base.file_sha(PRIOR_SELECTED/'public-index.sqlite'),
        'ancestor_summary_sha256':base.file_sha(ANCESTOR/'summary.json'),
        'all_ancestral_and_losing_candidate_aliases_preserved':True})
    freezer.STATE=FROZEN;freezer.PRIOR=proxy;freezer.PRIOR_SCORE=PRIOR_SCORE;freezer.ANCESTOR=ANCESTOR;freezer.main()


def prepare():
    STATE.mkdir(exist_ok=True)
    if (STATE/'candidate-input-files.json').exists():raise ValueError('candidate manifest already frozen')
    summary=json.loads((source.STATE/'normalize-summary.json').read_text())
    policy=json.loads((source.STATE/'normalization-policy.json').read_text())
    if summary['policy_sha256']!=base.file_sha(source.STATE/'normalization-policy.json'):
        raise ValueError('complete normalized source policy changed')
    if policy['driver_sha256']!=base.file_sha(Path(source.__file__)):raise ValueError('normalizer changed')
    files=[];count=0
    for part in summary['files']:
        if part['policy_sha256']!=summary['policy_sha256'] or base.file_sha(part['path'])!=part['sha256']:
            raise ValueError('normalized complete archive differs')
        count+=part['counts'].get('candidate_answers',0)
        if part['counts'].get('candidate_answers',0):files.append({'file_id':len(files),'path':part['path'],
            'sha256':part['sha256'],'format':archive.FORMAT,'wave':'wikidata-relations-v1'})
    if count!=summary['counts']['candidate_answers']:raise ValueError('candidate coverage differs')
    first=json.loads((source.diagnosis.STATE/'semantic-audit-v1/summary.json').read_text())
    base.save(STATE/'candidate-holds.private.json',{'record_ids':first['excluded_record_ids'],
        'query_hashes':policy['first_held_query_hashes'],'search_allowed':[],'supplemental_prior_queries':[]})
    base.save(STATE/'candidate-input-files.json',files)
    base.save(STATE/'candidate-policy-binding.json',{'driver_sha256':base.file_sha(Path(__file__)),
        'canonicalizer_sha256':base.file_sha(Path(candidates.__file__)),
        'structural_guard_algorithm_sha256':base.file_sha(Path(identity.__file__)),
        'storage_loader_sha256':base.file_sha(Path(archive.__file__)),
        'framed_index_adapter_sha256':base.file_sha(Path(framed.__file__)),
        'normalization_summary_sha256':base.file_sha(source.STATE/'normalize-summary.json'),
        'native_v26_policy_sha256':base.file_sha(candidates.POLICY),
        'source_scope':'One canonical exact target per eligible anchor; all singleton/group assertions must be checked against the complete index before selection.',
        'all_frames_hash_checked_by_index_reader':True,'no_model_requests_or_scores_claimed':True,
        'any_discovered_alternatives_require_native_v26_comparison':True,'all_first_holds_preserved':True})
    print(json.dumps({'files':len(files),'candidate_answers':count}),flush=True)


def configure():
    binding=json.loads((STATE/'candidate-policy-binding.json').read_text())
    for key,path in [('driver_sha256',Path(__file__)),('canonicalizer_sha256',Path(candidates.__file__)),
        ('structural_guard_algorithm_sha256',Path(identity.__file__)),('storage_loader_sha256',Path(archive.__file__)),
        ('framed_index_adapter_sha256',Path(framed.__file__))]:
        if base.file_sha(path)!=binding[key]:raise ValueError('bound source index implementation changed')
    candidates.STATE=STATE;candidates.index_file=framed.index_file
    identity.STATE=STATE/'episode-identity-v1';identity.job=framed.guard_file


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','prepare','index','guard']);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if not 1<=args.workers<=16:raise ValueError('worker bound')
    if args.phase=='freeze':freeze()
    elif args.phase=='prepare':prepare()
    else:
        configure()
        if args.phase=='index':
            if (STATE/'candidate-index-summary.json').exists():raise ValueError('index already complete')
            candidates.index(args.workers)
        else:identity.main()
