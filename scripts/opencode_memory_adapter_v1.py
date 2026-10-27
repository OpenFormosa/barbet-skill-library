"""Source-test-qualified OpenCode candidates; source programs remain inert text."""
import json
import math

from barbet_inference.memory import MemoryItem, canonical, compose_memory_prompt
from barbet_inference.sft import Rejected, RESERVED, augment_system, normalize_messages
from scripts import build_memory_sft_v2 as base

CRITERIA = {'requirement_conformance', 'logical_correctness', 'edge_case_consideration'}


def source_quality(raw):
    try:
        status = json.loads(raw['tests_execution_status']); tests = json.loads(raw['unit_tests'])
        judge = json.loads(raw['llm_judgement']); score = raw['average_test_score']
    except (ValueError, KeyError, TypeError) as exc:
        raise Rejected('malformed_source_verification_metadata') from exc
    if not isinstance(status, list) or not status or any(s != 'pass' for s in status):
        raise Rejected('source_test_failure_or_missing_status')
    if not isinstance(tests, list) or len(tests) != len(status) or any(not isinstance(t, str) or not t.strip() for t in tests):
        raise Rejected('source_test_definition_status_mismatch')
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score != 1.0:
        raise Rejected('source_test_score_inconsistent')
    if not isinstance(judge, dict) or set(judge) != CRITERIA:
        raise Rejected('unknown_source_judge_contract')
    scores = {}
    for key in sorted(CRITERIA):
        item = judge[key]
        value = item.get('score') if isinstance(item, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 4 <= value <= 5:
            raise Rejected('source_judge_score_below_four_or_invalid')
        scores[key] = value
    return {'upstream_test_cases': len(status), 'upstream_test_pass_fraction': score,
            'upstream_judge_scores': scores, 'source_tests_sha256': base.sha(raw['unit_tests']),
            'source_test_status_sha256': base.sha(raw['tests_execution_status']),
            'locally_executed': False, 'independently_verified_correctness': False}


def adapt(raw, source, index, skills):
    verification = source_quality(raw)
    question, answer = raw.get('input'), raw.get('output')
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
        raise Rejected('missing_coding_question_or_response')
    if RESERVED.search(question + answer): raise Rejected('literal_reserved_delimiter')
    if raw.get('domain') not in ['generic', 'algorithmic'] or raw.get('generation_algorithm') not in ['self-instruct', 'evol-instruct']:
        raise Rejected('unknown_opencode_generation_cohort')
    qh = base.sha(base.norm(question)); placement = 'front' if int(qh[:8], 16) % 2 else 'back'
    # Physical file/row attribution belongs to metadata, so the same task and
    # answer in different source shards cannot create different PPL conditions.
    source_url = f"https://huggingface.co/datasets/{source['repo']}/tree/{source['revision']}"
    item = MemoryItem('coding-task', question, source_url, observation_origin='user_task_specification')
    prompt = compose_memory_prompt('Complete the coding task in memory item coding-task. Follow its requirements and examples.', [item], placement=placement)
    messages = augment_system(normalize_messages([{'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': answer}], [], {}), skills[0])
    rid = base.sha(canonical({'version': 'memory-harness-sft-v2', 'messages': messages, 'tools': []}))
    family = base.sha('query-family:' + base.norm(question, numbers=True))
    return {'record_id': rid, 'family_id': family, 'source_family_id': family, 'template_family_id': family,
        'source_task_key': qh, 'query_hash': qh, 'split': 'unassigned', 'domain': 'coding',
        'source': source['repo'], 'upstream_source': source['repo'], 'license': 'cc-by-4.0',
        'messages': messages, 'tools': [], 'memory': {'documents': [item.__dict__]},
        'skills': skills[1], 'skill_ids': base.ROUTES['coding'],
        'verification_tier': 'upstream_tests_all_pass_and_judge_scores_at_least_four_not_reexecuted',
        'observation_origin': 'user_task_specification', 'memory_placement': placement,
        'actual_retrieval': False, 'recorded_error_followup': False, 'locally_executed_source_program': False,
        'assistant_turns': 1, 'tool_observations': 0,
        'source_refs': {'repo': source['repo'], 'revision': source['revision'], 'file': source['path'],
            'file_sha256': source['sha256'], 'row_index': index, 'upstream_id': raw['id'],
            'raw_row_sha256': base.sha(canonical(raw)), 'attribution_url': source_url, 'license': 'cc-by-4.0',
            'upstream_domain': raw['domain'], 'generation_algorithm': raw['generation_algorithm'],
            'verification': verification, 'posthoc_unit_tests_in_prompt': False, 'judge_comments_in_prompt': False,
            'new_semantic_generation': False,
            'transformations': ['complete original coding task as causal memory', 'source metadata independent of prompt condition',
                                'unchanged public response', 'skill prefill', 'assistant-only loss']}}
