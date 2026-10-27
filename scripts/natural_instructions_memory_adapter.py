"""Causal definition/example/input memory from pinned public NI task files.

Reference outputs remain alternatives for later v26 ranking, never prompt memory.
This is a source adapter, not a semantic verifier or a code executor.
"""
import hashlib
import re
import unicodedata

from barbet_inference.memory import MemoryItem, canonical, compose_memory_prompt
from barbet_inference.sft import Rejected, RESERVED

LICENSES = {'CC BY 4.0': 'cc-by-4.0', 'Apache 2.0': 'apache-2.0',
            'MIT': 'mit', 'CC BY-SA 4.0': 'cc-by-sa-4.0'}
EXCLUDED_CATEGORIES = {'Wrong Candidate Generation', 'Story Composition', 'Misc.'}


def norm(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value).casefold()).strip()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def task_contract(task):
    licenses = task.get('Instance License', [])
    if len(licenses) != 1 or licenses[0] not in LICENSES:
        raise Rejected('ni_instance_license_not_admitted')
    definitions = task.get('Definition')
    if not isinstance(definitions, list) or len(definitions) != 1 or not isinstance(definitions[0], str) or not definitions[0].strip():
        raise Rejected('ni_ambiguous_task_definition')
    categories = task.get('Categories', [])
    if not categories or set(categories) & EXCLUDED_CATEGORIES:
        raise Rejected('ni_category_outside_contextual_task_scope')
    return definitions[0], LICENSES[licenses[0]], categories


def exact_openorca_views(definition, target_input):
    """Known source wrappers only; no inferred paraphrase equivalence."""
    d, i = definition, target_input
    return [
        'You will be given a definition of a task first, then some input of the task.\n' + d + '\n\n' + i + '\nOutput:',
        'Given the task definition and input, reply with output. ' + d + '\n\n' + i + '\n',
        'Teacher:' + d + '\nTeacher: Now, understand the problem? Solve this instance: ' + i + '\nStudent:',
        'Instructions: ' + d + '\nInput: ' + i + '\nOutput:',
        'Definition: ' + d + '\nInput: ' + i + '\nOutput:',
        'Detailed Instructions: ' + d + '\nQ: ' + i + '\nA:',
        'Detailed Instructions: ' + d + '\nProblem:' + i + '\nSolution:',
        'Detailed Instructions: ' + d + '\nProblem: ' + i + '\nSolution:',
        'Q: ' + d + '\n' + i + '\nA:',
        d + '\n\n' + i,
        d + '\nQ: ' + i + '\nA: ',
    ]


def adapt(task, instance, *, source_url, max_examples=2):
    definition, license_name, categories = task_contract(task)
    target = instance.get('input'); outputs = instance.get('output')
    if not isinstance(target, str) or not target.strip() or not isinstance(outputs, list) or not outputs:
        raise Rejected('ni_missing_input_or_outputs')
    if any(not isinstance(a, str) or not a.strip() for a in outputs):
        raise Rejected('ni_invalid_reference_output')
    if RESERVED.search(definition + target + '\n'.join(outputs)):
        raise Rejected('ni_literal_reserved_delimiter')
    key = canonical({'definition': norm(definition), 'input': norm(target)})
    placement = 'front' if int(digest(key)[:8], 16) % 2 else 'back'
    items = []; examples = []; skipped = []; seen = {norm(target)}
    for n, example in enumerate(task.get('Positive Examples', [])):
        if len(items) >= max_examples:
            break
        if not isinstance(example.get('input'), str) or not isinstance(example.get('output'), str):
            raise Rejected('ni_invalid_worked_example')
        identity = norm(example['input'])
        if identity in seen:
            skipped.append(n); continue
        seen.add(identity)
        text = canonical({'input': example['input'], 'output': example['output'],
                          'explanation': example.get('explanation', '')})
        if RESERVED.search(text):
            raise Rejected('ni_reserved_delimiter_in_worked_example')
        items.append(MemoryItem(f'worked-example-{n}', text, source_url,
                                observation_origin='upstream_public_worked_example'))
        examples.append(n)
    items.append(MemoryItem('target-input', target, source_url))
    question = ('Apply this task definition to the memory item named target-input. '
                'The other memory items are worked examples of different inputs. '
                'Return the task output for target-input only, following the definition.\n\n'
                'TASK DEFINITION:\n' + definition)
    prompt = compose_memory_prompt(question, items, placement=placement)
    domain = 'coding' if set(categories) & {'Text to Code', 'Program Execution'} else 'reading'
    # This family links explicit shared passages across question-generation and
    # answering tasks, and identical source inputs across instruction tasks.
    family_input = re.split(r'\s+Question:\s*', target, maxsplit=1)[0] if target.startswith('Passage:') else target
    return {'definition': definition, 'target_input': target, 'query': definition + '\n\n' + target,
        'prompt': prompt, 'answers': list(dict.fromkeys(outputs)), 'source_task_key': digest(key),
        'family_key': digest('ni-original-input:' + norm(family_input)),
        'domain': domain, 'license': license_name, 'categories': categories,
        'memory_placement': placement, 'memory': {'documents': [item.__dict__ for item in items]},
        'worked_example_indices': examples, 'target_or_duplicate_example_indices_excluded': skipped,
        'target_answers_in_memory': False, 'actual_retrieval': False}
