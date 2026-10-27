"""Apply unchanged full candidate and episode guards to lexical source frames."""
import argparse
import json
from pathlib import Path
import shutil

from barbet_inference import candidate_archive as archive
from scripts import build_memory_sft_v2 as base
from scripts import normalize_conceptnet_relations_v1 as source
from scripts import prepare_v26_answer_candidates as candidates
from scripts import guard_v26_episode_identity as identity
from scripts import framed_candidate_io_v1 as framed

STATE=source.STATE/'candidate-selection-v1'


def reserve(stage,amount):
    resources=json.loads(source.RESOURCES.read_text())['resources'];used=source.processed_bytes();free=shutil.disk_usage(base.WORK).free
    remaining=0
    if not (source.GRAPH_SELECTED/'compile-summary.json').exists():
        graph=json.loads((source.GRAPH_SELECTED/'framed-compile-policy.json').read_text())
        written=sum(p.stat().st_size for p in source.GRAPH_OUT.rglob('*.parquet'))
        remaining=max(0,graph['reserved_compile_bytes']-written)
    receipt={'stage':stage,'reserved_stage_bytes':amount,'concurrent_graph_remaining_bytes':remaining,
        'processed_bytes':used,'free_bytes':free,'resource_policy_sha256':base.file_sha(source.RESOURCES),
        'passed':used+amount+remaining<=resources['new_processed_cap_bytes'] and free>=resources['minimum_free_disk_bytes']+amount+remaining}
    dest=STATE/(stage+'-resource-preflight.json')
    if not dest.exists():base.save(dest,receipt)
    if not receipt['passed']:raise ValueError('combined lexical/graph stage resource bound')


def prepare():
    STATE.mkdir(exist_ok=True)
    if (STATE/'candidate-input-files.json').exists():raise ValueError('lexical index manifest already frozen')
    summary=json.loads((source.STATE/'normalize-summary.json').read_text())
    policy=json.loads((source.STATE/'normalization-policy.json').read_text())
    if summary['policy_sha256']!=base.file_sha(source.STATE/'normalization-policy.json') or policy['driver_sha256']!=base.file_sha(Path(source.__file__)):
        raise ValueError('lexical source policy changed')
    files=[];count=0
    for part in summary['files']:
        if part['policy_sha256']!=summary['policy_sha256'] or base.file_sha(part['path'])!=part['sha256']:
            raise ValueError('complete lexical archive differs')
        count+=part['counts'].get('candidate_answers',0)
        if part['counts'].get('candidate_answers',0):files.append({'file_id':len(files),'path':part['path'],
            'sha256':part['sha256'],'format':archive.FORMAT,'wave':'conceptnet-lexical-relations-v1'})
    if count!=summary['counts']['candidate_answers']:raise ValueError('complete source coverage differs')
    first=json.loads((source.diagnosis.STATE/'semantic-audit-v1/summary.json').read_text())
    base.save(STATE/'candidate-holds.private.json',{'record_ids':first['excluded_record_ids'],
        'query_hashes':policy['first_held_query_hashes'],'search_allowed':[],'supplemental_prior_queries':[]})
    base.save(STATE/'candidate-input-files.json',files)
    base.save(STATE/'candidate-policy-binding.json',{'driver_sha256':base.file_sha(Path(__file__)),
        'canonicalizer_sha256':base.file_sha(Path(candidates.__file__)),
        'episode_guard_sha256':base.file_sha(Path(identity.__file__)),
        'archive_reader_sha256':base.file_sha(Path(archive.__file__)),
        'framed_adapter_sha256':base.file_sha(Path(framed.__file__)),
        'normalization_summary_sha256':base.file_sha(source.STATE/'normalize-summary.json'),
        'native_v26_policy_sha256':base.file_sha(candidates.POLICY),
        'all_frames_fully_revalidated_by_index_and_identity_readers':True,
        'any_discovered_alternative_answers_require_frozen_native_v26_scores':True,
        'prior_aliases_referenced_without_copying_or_mutating_frozen_databases':True,
        'all_first_holds_preserved':True,'no_PPL_values_or_model_requests_claimed':True})
    print(json.dumps({'files':len(files),'candidate_answers':count}),flush=True)


def configure():
    policy=json.loads((STATE/'candidate-policy-binding.json').read_text())
    for key,path in [('driver_sha256',Path(__file__)),('canonicalizer_sha256',Path(candidates.__file__)),
        ('episode_guard_sha256',Path(identity.__file__)),('archive_reader_sha256',Path(archive.__file__)),('framed_adapter_sha256',Path(framed.__file__))]:
        if base.file_sha(path)!=policy[key]:raise ValueError('frozen lexical index implementation changed')
    candidates.STATE=STATE;candidates.index_file=framed.index_file
    identity.STATE=STATE/'episode-identity-v1';identity.job=framed.guard_file


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','index','guard']);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if not 1<=args.workers<=8:raise ValueError('concurrent CPU worker bound')
    if args.phase=='prepare':prepare()
    else:
        configure();reserve(args.phase,(3 if args.phase=='index' else 2)*1024**3)
        if args.phase=='index':
            if (STATE/'candidate-index-summary.json').exists():raise ValueError('complete index already frozen')
            candidates.index(args.workers)
        else:identity.main()
