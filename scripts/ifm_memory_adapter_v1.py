"""Conservative contextual projection of public IFM final answers/actions.

Thinking fields are excluded; original tool schemas and observations are kept.
This adapter performs structural checks, not independent semantic verification.
"""
import copy
import re

from barbet_inference.memory import MemoryItem, canonical, compose_memory_prompt
from barbet_inference.sft import Rejected, RESERVED, augment_system, normalize_messages, prepare_tools
from barbet_inference.tool_contract import trajectory_has_undeclared_arguments
from scripts import build_memory_sft_v2 as base
from scripts.inventory_ifm_context_v1 import project
from scripts.normalize_opencode_memory_v1 import syntax_gate

IDENTITY_ONLY = {
    'You are K2, a helpful assistant created by Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) Institute of Foundation Models (IFM).',
}
CREDENTIAL = re.compile(r'(?:^|[_-])(?:api[_-]?key|access[_-]?token|auth(?:orization)?(?:[_-]?(?:key|token))?|authentication[_-]?(?:key|token)|password|secret)(?:$|[_-])', re.I)
PLACEHOLDER = re.compile(r'your[_ -].*(?:key|token|password)|(?:insert|replace)[_ -].*(?:key|token|password)|<[^>]*(?:key|token|password)[^>]*>', re.I)
EXTERNAL_FIGURE = re.compile(r'\b(?:diagram|figure|image|picture|graph)\b', re.I)
NONSENSE_TEMPLATE = re.compile(r'\b(?:for Engineer for Professionals|(?:Extreme|High|Low) (?:Ear|Nose|Eye|Foot|Hand|Oral) hygiene)\b', re.I)


def credential_gate(messages, tools):
    earlier = canonical(tools)
    for message in messages:
        if message['role'] == 'assistant':
            for call in message.get('tool_calls', []):
                for key, value in call['function']['arguments'].items():
                    if not CREDENTIAL.search(key) or value in ('', None): continue
                    if not isinstance(value, str) or PLACEHOLDER.search(value) or value not in earlier:
                        raise Rejected('credential_value_not_supplied_or_placeholder')
        # Previous assistant guesses do not establish a credential as a fact.
        elif message['role'] in {'user', 'system', 'tool'}:
            earlier += '\n' + message['content']


def adapt(raw, source, index, skills):
    category = source['path'].split('/')[0]
    if category not in {'agent', 'code', 'math'}: raise Rejected('source_category_not_in_prospective_scope')
    try: projected, question = project(raw)
    except (TypeError, ValueError, KeyError) as exc: raise Rejected('invalid_visible_source_projection') from exc
    if '<solution>' in canonical(projected[:-1]): raise Rejected('reference_solution_in_input')
    if RESERVED.search(canonical(projected)): raise Rejected('literal_reserved_delimiter')
    source_projection_sha = base.sha(canonical(projected))
    projected = copy.deepcopy(projected); raw_tools = []; seen_tools = set(); removed_identity = []
    for m in projected:
        if m.get('role') == 'system' and m.get('content') in IDENTITY_ONLY:
            removed_identity.append(base.sha(m['content'])); m['content'] = ''
        if m.get('tools'):
            if m['role'] != 'system': raise Rejected('tool_schema_outside_system')
            if not isinstance(m['tools'], list): raise Rejected('tool_registry_not_array')
            for tool in m['tools']:
                key = canonical(tool)
                if key not in seen_tools: raw_tools.append(tool); seen_tools.add(key)
    tools, validators = prepare_tools(raw_tools)
    messages = normalize_messages(projected, tools, validators)
    if trajectory_has_undeclared_arguments(messages, tools): raise Rejected('undeclared_root_function_argument')
    credential_gate(messages, tools)
    domain = {'code': 'coding', 'math': 'math', 'agent': 'tool_use'}[category]
    if category == 'agent' and not tools: raise Rejected('agent_without_explicit_source_registry')
    if category == 'math' and EXTERNAL_FIGURE.search(question): raise Rejected('external_math_figure_not_rendered')
    static_blocks = None
    if category == 'code':
        if NONSENSE_TEMPLATE.search(question): raise Rejected('incoherent_tiny_codes_instruction_template')
        static_blocks = syntax_gate(messages[-1]['content'])
    qh = base.sha(base.norm(question)); placement = 'front' if int(qh[:8], 16) % 2 else 'back'
    url = f"https://huggingface.co/datasets/{source['repo']}/tree/{source['revision']}"
    item = MemoryItem('initial-task', question, url, observation_origin='user_task_specification')
    first_user = next(i for i, m in enumerate(messages) if m['role'] == 'user')
    if messages[first_user]['content'] != question: raise Rejected('source_first_query_changed_by_normalizer')
    messages[first_user]['content'] = compose_memory_prompt(
        'Complete the initial task in memory item initial-task. Preserve later user requirements and tool observations.',
        [item], placement=placement)
    messages = augment_system(messages, skills[domain][0])
    rid = base.sha(canonical({'version': 'memory-harness-sft-v2', 'messages': messages, 'tools': tools}))
    family = base.sha('query-family:' + base.norm(question, numbers=True))
    observations = sum(m['role'] == 'tool' for m in messages)
    return {'record_id': rid, 'query_hash': qh, 'family_id': family, 'source_family_id': family,
        'template_family_id': family, 'source_task_key': qh, 'split': 'unassigned', 'domain': domain,
        'source': source['repo'], 'upstream_source': source['repo'], 'license': 'cc-by-4.0',
        'messages': messages, 'tools': tools, 'memory': {'documents': [item.__dict__]},
        'skills': skills[domain][1], 'skill_ids': base.ROUTES[domain],
        'verification_tier': 'public_regenerated_response_structure_checked_not_locally_semantic_verified',
        'observation_origin': 'upstream_recorded_or_simulated' if observations else 'user_task_specification',
        'memory_placement': placement, 'actual_retrieval': False,
        'recorded_error_followup': False, 'locally_executed_source_program': False,
        'assistant_turns': sum(m['role'] == 'assistant' and m['loss'] for m in messages),
        'tool_observations': observations,
        'source_refs': {'repo': source['repo'], 'revision': source['revision'], 'file': source['path'],
            'file_sha256': source['sha256'], 'row_index': index,
            'visible_source_projection_sha256': source_projection_sha,
            'source_projection_excludes_reasoning': True, 'attribution_url': url,
            'source_category': category, 'source_effort': source['path'].split('/')[-1].split('-')[0],
            'source_effort_not_a_new_question': True, 'license': 'cc-by-4.0',
            'upstream_row_level_original_dataset_not_provided': True,
            'removed_exact_identity_only_system_hashes': removed_identity,
            'python_static_blocks': static_blocks, 'source_code_executed': False,
            'new_semantic_generation': False,
            'transformations': ['exclude separate think fields', 'remove exact generic K2 identity-only system line',
                'retain source task, scenarios, declared schemas, actions and observations',
                'complete initial task as causal memory', 'skill prefill', 'assistant-only loss']}}
