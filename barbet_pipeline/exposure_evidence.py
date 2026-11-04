"""Report the actual TRAIN supervision mix without relabeling source quality."""
from collections import Counter, defaultdict


GRAPH_SOURCES = {'Wikidata5M/original-inductive-train', 'ConceptNet/5.7-original-lexical-english'}
QUALITY_SCOPES = {'verified', 'source_reference', 'reference_only'}


def summarize_training_exposure(statistics):
    if not statistics.get('all_exported_rows_measured'):
        raise ValueError('Complete corpus statistics are required')
    quality, source, domain = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    graph, non_graph_scoped = Counter(), Counter()
    for key, values in statistics['exposure_groups'].items():
        status, source_name, domain_name, split = key.split('|')
        if status not in QUALITY_SCOPES:
            raise ValueError('Unknown quality scope')
        if split != 'train':
            continue
        if set(values) != {'rows', 'input_tokens', 'supervised_tokens'}:
            raise ValueError('Unexpected exposure accounting fields')
        if any(not isinstance(v, int) or v < 0 for v in values.values()):
            raise ValueError('Invalid exposure count')
        for target in [quality[status], source[status + '|' + source_name], domain[domain_name]]:
            target.update(values)
        if source_name in GRAPH_SOURCES:
            if status != 'verified':
                raise ValueError('Finite-graph source scope unexpectedly changed')
            graph.update(values)
        elif status != 'reference_only':
            non_graph_scoped.update(values)
    total = sum((v for v in quality.values()), Counter())
    supervised = total['supervised_tokens']
    scoped = quality['verified']['supervised_tokens'] + quality['source_reference']['supervised_tokens']
    if not supervised or not scoped:
        raise ValueError('Cannot characterize empty training exposure')
    return {'schema': 'barbet-train-exposure-evidence-v1', 'split': 'train', 'all_train_rows_measured': True,
        'totals': dict(total), 'by_quality_scope': {k: dict(v) for k, v in sorted(quality.items())},
        'by_source_and_scope': {k: dict(v) for k, v in sorted(source.items())},
        'by_inherited_domain_tag': {k: dict(v) for k, v in sorted(domain.items())},
        'finite_graph': dict(graph), 'non_graph_with_verification_or_source_reference': dict(non_graph_scoped),
        'reference_only_supervised_fraction': quality['reference_only']['supervised_tokens'] / supervised,
        'finite_graph_supervised_fraction_all_train': graph['supervised_tokens'] / supervised,
        'finite_graph_supervised_fraction_of_verified_plus_source_reference': graph['supervised_tokens'] / scoped,
        'interpretation': [
            'reference_only marks an independent-verification gap; it does not mean every target is wrong.',
            'Source fidelity and finite-graph semantic correctness are distinct verification scopes.',
            'Inherited math/coding/reading/tool labels do not establish semantic difficulty or independent correctness.',
            'A single global NLL is dominated by sources with more supervised tokens and cannot certify a balanced course.',
            'No row is removed, promoted or reselected by this report; validation/DEV/TEST scores do not select TRAIN.'
        ],
        'curriculum_constraints': {
            'uniform_full_corpus_sampling_validated': False,
            'uniform_scoped_subset_sampling_validated': False,
            'reference_only_automatic_training_admission': False,
            'fifty_fifty_general_semantic_mix_validated': False,
            'protect_verified_semantic_pairs_and_value_aligned_civic_examples_from_score_only_dropping': True,
            'broad_general_math_coding_instruction_verification_and_retention_evidence_still_required': True,
        }}
