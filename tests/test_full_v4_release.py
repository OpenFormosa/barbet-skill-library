import json
from pathlib import Path
import pytest
from barbet_pipeline.common import save_new, sha_file
from barbet_pipeline.full_release import collect_complete_release, configuration_specs, exposure_policy


def fixture_release(tmp_path):
    counts = {'active_rows_processed': 3, 'exported': 2, 'quarantined': 1,
              'status:verified': 1, 'status:source_reference': 0, 'status:reference_only': 1,
              'input_tokens': 20, 'supervised_tokens': 4, 'decision_points': 2}
    files = [{'remote': f'full-v4/{s}/records/train-one.parquet', 'local': '/not-used',
              'sha256': s, 'bytes': 10, 'rows': 1} for s in ['verified','reference_only']]
    files += [{**f, 'remote':f['remote'].replace('/records/','/tokenized/')} for f in files]
    files.append({'remote': 'full-v4/dispositions/one.parquet', 'local': '/not-used', 'sha256': 'd', 'bytes': 11, 'rows': 3})
    receipt={'source_path':'source-one','source_sha256':'s','token_source_sha256':'t','driver_sha256':'c',
             'quality_driver_sha256':'q','counts':counts,'by_source':{'processed:source':3},'files':files}
    save_new(tmp_path/'plan.json',{'jobs':[{'path':'source-one','sha256':'s','token_sha256':'t'}],
                                 'driver_sha256':'c','quality_driver_sha256':'q','active_rows_required':3})
    save_new(tmp_path/'receipts/one.json',receipt)
    save_new(tmp_path/'processing-summary.json',{'counts':counts,'by_source':receipt['by_source'],'source_shards':1})
    save_new(tmp_path/'statistics-summary.json',{'source_shards':1,'counts':{'rows':2,'input_tokens':20,'supervised_tokens':4,'decision_points':2}})
    save_new(tmp_path/'statistics/one.json',{'source_receipt_sha256':sha_file(tmp_path/'receipts/one.json')})
    save_new(tmp_path/'source-visibility/one.json',{'source_receipt_sha256':sha_file(tmp_path/'receipts/one.json')})
    save_new(tmp_path/'source-visibility-summary.json',{'source_shards':1,'source_counts':{},'all_exported_QA_NI_visible_prefixes_reconstructed':True})
    save_new(tmp_path/'graph-visibility/one.json',{'source_receipt_sha256':sha_file(tmp_path/'receipts/one.json')})
    save_new(tmp_path/'graph-visibility-summary.json',{'source_shards':1,'source_counts':{},'all_exported_graph_queries_and_visible_state_checked':True})
    save_new(tmp_path/'actual-changes/one.json',{'source_receipt_sha256':sha_file(tmp_path/'receipts/one.json')})
    save_new(tmp_path/'actual-changes-summary.json',{'source_shards':1,'all_exported_rows_classified':True,
        'counts':{'exported_rows':2,'actual_serialized_text_changed':1,'actual_serialized_text_identical':1}})
    save_new(tmp_path/'publication/groups/one.json',{'all_anonymous_hashes_match':True,'verified_count':5,'files':files})
    save_new(tmp_path/'publication/readback/one.json',{'verified':[
        {'path':f['remote'],'sha256':f['sha256'],'bytes':f['bytes'],'anonymous':True} for f in files]})
    return files


def mutate(path, fn):
    value=json.loads(path.read_text());fn(value);path.write_text(json.dumps(value))


def test_complete_accounting_requires_all_verified_files(tmp_path):
    expected=fixture_release(tmp_path)
    summary, statistics, files=collect_complete_release(tmp_path)
    assert files==expected and summary['counts']['active_rows_processed']==3


@pytest.mark.parametrize('file,change',[
    ('plan.json',lambda x:x['jobs'].append({'path':'missing','sha256':'s','token_sha256':'t'})),
    ('receipts/one.json',lambda x:x.update(driver_sha256='changed-code')),
    ('processing-summary.json',lambda x:x['counts'].update(exported=3)),
    ('statistics-summary.json',lambda x:x['counts'].update(rows=1)),
    ('publication/readback/one.json',lambda x:x['verified'][0].update(anonymous=False)),
    ('publication/readback/one.json',lambda x:x['verified'][0].update(sha256='wrong')),
    ('publication/groups/one.json',lambda x:x.update(verified_count=2)),
])
def test_incomplete_or_changed_release_is_refused(tmp_path,file,change):
    fixture_release(tmp_path);mutate(tmp_path/file,change)
    with pytest.raises(ValueError):collect_complete_release(tmp_path)


def test_reference_config_cannot_enter_verified_config(tmp_path):
    files=fixture_release(tmp_path)
    configs=configuration_specs(files)
    verified=next(x for x in configs if x['config_name']=='full_v4_verified_records')
    assert all('/verified/' in p for s in verified['data_files'] for p in s['path'])
    default=next(x for x in configs if x.get('default'))
    assert default['config_name']=='full_v4_dispositions'
    assert default['data_files'][0]['split']=='corpus'


def test_exposure_policy_never_calls_source_fidelity_training_success():
    policy=exposure_policy({'exposure_groups':{'source_reference|source|reading|train':{'rows':100,'input_tokens':200,'supervised_tokens':10}}})
    assert not policy['automatic_reference_only_admission']
    assert not policy['proposed_general_semantic_supervised_token_mix']['validated']
    assert policy['measured_exposure_groups'][0]['supervised_tokens']==10
    assert policy['max_views_per_parent_record_per_pass']==1
    assert policy['do_not_count_legacy_family_ids_as_independent_question_roots']
