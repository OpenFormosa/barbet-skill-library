"""Fail-closed final accounting for the exhaustive v4 data release."""
from collections import Counter
from pathlib import Path
from .common import read_json, sha_file


def collect_complete_release(state):
    state = Path(state)
    plan = read_json(state / 'plan.json')
    summary = read_json(state / 'processing-summary.json')
    statistics = read_json(state / 'statistics-summary.json')
    expected = {j['path']: j for j in plan['jobs']}
    amendment_path = state / 'ranked-input-amendment.json'
    amendment = read_json(amendment_path) if amendment_path.exists() else None
    if amendment and amendment['initial_driver_sha256'] != plan['driver_sha256']:
        raise ValueError('Code amendment does not derive from the frozen driver')
    if len(expected) != len(plan['jobs']):
        raise ValueError('Duplicate planned source shard')
    receipts = [read_json(p) for p in sorted((state / 'receipts').glob('*.json'))]
    actual = {r['source_path']: r for r in receipts}
    if set(expected) != set(actual) or len(actual) != len(receipts):
        raise ValueError('Not every planned source shard has exactly one receipt')
    counts, sources = Counter(), Counter()
    files = {}
    for name, receipt in actual.items():
        job = expected[name]
        if (receipt['source_sha256'] != job['sha256'] or
                receipt['token_source_sha256'] != job['token_sha256']):
            raise ValueError('Source receipt identity differs from frozen plan')
        expected_driver = plan['driver_sha256']
        if amendment and name not in amendment['initial_successful_source_paths']:
            expected_driver = amendment['corrected_driver_sha256']
        if (receipt['driver_sha256'] != expected_driver or
                receipt['quality_driver_sha256'] != plan['quality_driver_sha256']):
            raise ValueError('Processing code changed after the frozen plan')
        counts.update(receipt['counts'])
        sources.update(receipt['by_source'])
        for f in receipt['files']:
            if f['remote'] in files:
                raise ValueError('Duplicate output path')
            files[f['remote']] = f
    if dict(counts) != summary['counts'] or dict(sources) != summary['by_source']:
        raise ValueError('Aggregate processing counts do not match the source receipts')
    if counts['active_rows_processed'] != plan['active_rows_required']:
        raise ValueError('Incomplete active corpus')
    if counts['exported'] + counts['quarantined'] != counts['active_rows_processed']:
        raise ValueError('Unaccounted source row')
    if sum(counts['status:' + x] for x in ['verified', 'source_reference', 'reference_only']) != counts['exported']:
        raise ValueError('Unknown accepted status')
    if summary['source_shards'] != len(expected) or statistics['source_shards'] != len(expected):
        raise ValueError('Missing complete statistics or processing shards')
    for key in ['input_tokens', 'supervised_tokens', 'decision_points']:
        if statistics['counts'][key] != counts[key]:
            raise ValueError('Independent metadata scan disagrees: ' + key)
    if statistics['counts']['rows'] != counts['exported']:
        raise ValueError('Independent metadata scan missed rows')
    record_files = [f for f in files.values() if '/records/' in f['remote']]
    token_files = [f for f in files.values() if '/tokenized/' in f['remote']]
    disposition_files = [f for f in files.values() if '/dispositions/' in f['remote']]
    if sum(f['rows'] for f in record_files) != counts['exported']:
        raise ValueError('Saved record file rows do not match the complete export')
    if sum(f['rows'] for f in disposition_files) != counts['active_rows_processed']:
        raise ValueError('Saved dispositions do not cover the complete active corpus')
    if len(record_files) != len(token_files):
        raise ValueError('Record and tokenized shard coverage differs')
    for f in record_files:
        partner = files.get(f['remote'].replace('/records/', '/tokenized/'))
        if partner is None or partner['rows'] != f['rows']:
            raise ValueError('Record and tokenized shard rows differ')
    published = {}
    for path in sorted((state / 'publication/groups').glob('*.json')):
        group = read_json(path)
        if not group['all_anonymous_hashes_match'] or group['verified_count'] != len(group['files']):
            raise ValueError('Incomplete anonymous byte readback')
        for f in group['files']:
            if f['remote'] in published:
                raise ValueError('Output appears in two publication groups')
            published[f['remote']] = f
    if published != files:
        raise ValueError('Published files differ from complete processing receipts')
    readback = {}
    for path in sorted((state / 'publication/readback').glob('*.json')):
        for proof in read_json(path)['verified']:
            if proof['path'] in readback:
                raise ValueError('Repeated anonymous readback proof')
            if not proof['anonymous']:
                raise ValueError('Readback used authenticated access')
            readback[proof['path']] = proof
    if set(readback) != set(files):
        raise ValueError('Missing file-level anonymous readback')
    for name, item in files.items():
        proof = readback[name]
        if (item['sha256'], item['bytes']) != (proof['sha256'], proof['bytes']):
            raise ValueError('Anonymous byte proof mismatch')
    stat_paths = list((state / 'statistics').glob('*.json'))
    if len(stat_paths) != len(expected):
        raise ValueError('Not all source statistics receipts are present')
    for path in stat_paths:
        if read_json(path)['source_receipt_sha256'] != sha_file(state / 'receipts' / path.name):
            raise ValueError('Statistics source receipt changed')
    visibility = read_json(state / 'source-visibility-summary.json')
    if (visibility['source_shards'] != len(expected) or
            not visibility['all_exported_QA_NI_visible_prefixes_reconstructed']):
        raise ValueError('Complete QA/NI model-visible context audit is missing')
    for source in ['rajpurkar/squad_v2','hotpotqa/hotpot_qa','allenai/natural-instructions']:
        required = sum(sources[status + ':' + source] for status in ['verified','source_reference'])
        if visibility['source_counts'].get(source,0) != required:
            raise ValueError('QA/NI visible context coverage mismatch: ' + source)
    visibility_paths = list((state / 'source-visibility').glob('*.json'))
    if len(visibility_paths) != len(expected):
        raise ValueError('Missing model-visible context receipt')
    for path in visibility_paths:
        if read_json(path)['source_receipt_sha256'] != sha_file(state / 'receipts' / path.name):
            raise ValueError('Model-visible context source receipt changed')
    graph = read_json(state / 'graph-visibility-summary.json')
    expected_graph = {k[len('verified:'):]:v for k,v in sources.items()
                      if k.startswith(('verified:Wikidata5M/','verified:ConceptNet/'))}
    if (graph['source_shards'] != len(expected) or not graph['all_exported_graph_queries_and_visible_state_checked']
            or graph['source_counts'] != expected_graph):
        raise ValueError('Graph visible operation/state coverage is incomplete')
    graph_paths = list((state / 'graph-visibility').glob('*.json'))
    if len(graph_paths) != len(expected):raise ValueError('Missing graph input audit receipt')
    for path in graph_paths:
        if read_json(path)['source_receipt_sha256'] != sha_file(state / 'receipts' / path.name):
            raise ValueError('Graph visibility source receipt changed')
    changes = read_json(state / 'actual-changes-summary.json')
    if changes['source_shards'] != len(expected) or not changes['all_exported_rows_classified']:
        raise ValueError('Actual text-change census is incomplete')
    cc = changes['counts']
    if (cc['exported_rows'] != counts['exported'] or
            cc['actual_serialized_text_changed'] + cc['actual_serialized_text_identical'] != counts['exported']):
        raise ValueError('Actual text changes do not account for all exported rows')
    changes_paths = list((state / 'actual-changes').glob('*.json'))
    if len(changes_paths) != len(expected):raise ValueError('Missing actual-change census receipt')
    for path in changes_paths:
        if read_json(path)['source_receipt_sha256'] != sha_file(state / 'receipts' / path.name):
            raise ValueError('Actual-change census source receipt changed')
    return summary, statistics, list(files.values())


def configuration_specs(files):
    """Use exact paths; reference-only and inventory cannot enter TRAIN by glob."""
    configs = []
    for status in ['verified', 'source_reference', 'reference_only']:
        for kind in ['records', 'tokenized']:
            entries = []
            for split in ['train', 'validation']:
                paths = sorted(f['remote'] for f in files
                               if f['remote'].startswith(f'full-v4/{status}/{kind}/{split}-'))
                if paths:
                    entries.append({'split': split, 'path': paths})
            if entries:
                configs.append({'config_name': f'full_v4_{status}_{kind}', 'data_files': entries})
    configs.append({'config_name': 'full_v4_dispositions', 'default': True, 'data_files': [
        {'split': 'corpus', 'path': sorted(f['remote'] for f in files
                                          if f['remote'].startswith('full-v4/dispositions/'))}]})
    return configs


def exposure_policy(statistics):
    """An honest data menu, not a claimed experimentally validated mixture."""
    groups = []
    for key, values in sorted(statistics['exposure_groups'].items()):
        status, source, domain, split = key.split('|')
        groups.append({'status': status, 'source': source, 'domain': domain, 'split': split, **values})
    return {
        'status': 'DATA_MENU_NOT_VALIDATED_TRAINING_RECIPE',
        'default_configuration_is_audit_inventory_not_train': True,
        'automatic_reference_only_admission': False,
        'quality_status_is_not_learnability_or_retention': True,
        'never_concatenate_records_and_tokenized_as_distinct_examples': True,
        'do_not_uniformly_sample_the_full_pool': True,
        'reason': 'Source and operation exposure remain imbalanced; most verified rows are finite graph tasks.',
        'proposed_general_semantic_supervised_token_mix': {'general': 0.5, 'semantic': 0.5, 'validated': False},
        'max_views_per_parent_record_per_pass': 1,
        'root_task_id_semantics': 'conservative inherited family/split group; may contain several distinct questions',
        'distinct_question_identity': ['source', 'query_hash'],
        'do_not_count_legacy_family_ids_as_independent_question_roots': True,
        'unmet_recipe_requirements': [
            'independently verified general math, coding and natural-instruction content coverage',
            'student learnability, unseen-composition transfer and same-checkpoint retention evidence'],
        'measured_exposure_groups': groups,
    }
