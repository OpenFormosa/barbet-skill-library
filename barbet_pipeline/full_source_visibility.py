"""Reconstruct actual QA/NI decision prefixes from pinned source documents."""
import json
from .common import digest
from .full_quality import source_table, ni_task
from .factory import library_card
from .visibility import actor_messages
from barbet_inference.sft import normalize_messages, prepare_tools
from scripts.build_memory_sft_v2 import qa_task
from scripts.natural_instructions_memory_adapter import adapt

SOURCES={'rajpurkar/squad_v2','hotpotqa/hotpot_qa','allenai/natural-instructions'}


def check_source_visibility(row):
    source=row['source']
    if source not in SOURCES:return None
    refs=json.loads(row['source_refs_json']);messages=json.loads(row['messages_json'])
    memory=json.loads(row['memory_json']);tools=json.loads(row['tools_json'])
    if source=='allenai/natural-instructions':
        task=ni_task(refs['file'],refs['file_sha256']);instance=task['Instances'][refs['row_index']]
        view=adapt(task,instance,source_url=refs['attribution_url'])
        expected=normalize_messages([{'role':'user','content':view['prompt']},{'role':'assistant','content':messages[-1]['content']}],[],{})
        actual=[m for m in messages if m['role']!='system']
        if actual!=expected or memory!=view['memory'] or tools:
            raise ValueError('NI visible task definition/examples/target input differ from source reconstruction')
        return 'exact_NI_visible_definition_examples_and_target_input'
    raw_rows,provenance=source_table(source,refs['file']);raw=raw_rows[refs['row_index']]
    if row['natural_answer_rewritten']:
        if source!='rajpurkar/squad_v2':raise ValueError('Unexpected natural replacement source')
        root=digest([source,provenance['revision'],raw['id']])
        public={'task':raw['question']+'\nAnswer in English using only the supplied passage. If it does not support an answer, state that the passage is insufficient.',
            'context':[{'id':'source-passage','title':raw['title'],'revision':provenance['revision'],'text':raw['context']}],
            'skills':[library_card('grounded-evidence-answer')] if int(root[:2],16)%2==0 else [],
            'tools':[],'memory_placement':row['memory_placement']}
        if messages[:-1]!=actor_messages(public) or memory!={'documents':public['context']} or tools:
            raise ValueError('Natural SQuAD visible question/passage/skill differs from source reconstruction')
        return 'exact_natural_SQuAD_visible_question_and_passage'
    query,raw_messages,raw_tools,expected_memory,group,placement,retrieval=qa_task(raw,source)
    expected_tools,validators=prepare_tools(raw_tools)
    expected=normalize_messages(raw_messages,expected_tools,validators)
    actual=[m for m in messages if m['role']!='system']
    # Source target correctness is separately checked. This check deliberately
    # ends before the final answer, including every actual retrieval observation.
    if actual[:-1]!=expected[:-1] or memory!=expected_memory or tools!=expected_tools:
        raise ValueError('QA visible question/documents/retrieval observations differ from pinned source reconstruction')
    return 'exact_QA_visible_question_documents_and_retrieval_observations'
