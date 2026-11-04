"""Executable semantic curriculum specifications, independent of model wording.

Roots are procedural task instances, not claims of independent natural sources.
No benchmark item is imported. Private specifications never enter actor prompts.
"""
from __future__ import annotations

import copy
import random
from functools import lru_cache
from .common import canonical, digest
from .executors import Executor, StateLedger, tool_schema
from .oracle import solve, verify_execution, require_discriminating
from .visibility import actor_messages


KINDS = ['comparison', 'ordering', 'binding', 'units', 'rule_version', 'evidence_state', 'operation_state']
PHRASES = {'gt': '嚴格大於', 'ge': '大於或等於', 'lt': '嚴格小於', 'le': '小於或等於',
           'nth_smallest': '由小到大第', 'nth_largest': '由大到小第',
           'top_smallest': '由小到大的前', 'top_largest': '由大到小的前'}


@lru_cache(maxsize=1)
def skill_library():
    from barbet_inference.library import SkillLibrary
    return SkillLibrary()


def library_card(name):
    library = skill_library()
    definition = library.get(name)
    text, _ = library.render([name])
    return {'id':name, 'version':definition['version'], 'text':text}


def make_root(split, ordinal):
    if split not in {'train', 'dev', 'calibration', 'sealed_test'} or type(ordinal) is not int or ordinal < 0:
        raise ValueError('Invalid root allocation')
    root_id = digest(['semantic-curriculum-v3', split, ordinal])
    rng = random.Random(root_id)
    kind = KINDS[ordinal % len(KINDS)]
    table_id, field = 'table_' + root_id[:8], ['score', 'duration', 'capacity', 'rate'][ordinal % 4]
    threshold = rng.randint(-20, 30)
    values = [threshold - 9, threshold - 2, threshold, threshold, threshold + 3, threshold + 11]
    rng.shuffle(values)
    rows = [{'id': root_id[8:12] + '_' + chr(65+i), field: str(n),
             'source_id': 'SRC-' + root_id[12:20] + '-' + str(i),
             'destination_id': 'DST-' + root_id[20:28] + '-' + str(i),
             'optional': [None, False, 0, '0', '00017', True][i]} for i, n in enumerate(values)]
    if kind in {'ordering','rule_version'}:
        # Ties affect the selected rank, not just unrelated middle rows.
        ranked_values=[threshold-9,threshold-9,threshold-2,threshold+3,threshold+11,threshold+11]
        rng.shuffle(ranked_values)
        for row,value in zip(rows,ranked_values):row[field]=str(value)
    spec = {'kind': kind, 'rows': rows, 'table_id': table_id, 'field': field,
            'memory_placement':'front' if ordinal%2 else 'back'}
    if kind == 'comparison':
        spec.update(operation='gt' if ordinal % 2 else 'lt', threshold=str(threshold))
    elif kind == 'ordering':
        spec.update(operation='nth_smallest', k=1 + ordinal % 2)
    elif kind == 'binding':
        spec.update(row_id=rows[ordinal % len(rows)]['id'], field='source_id' if ordinal % 2 else 'optional')
    elif kind == 'units':
        for r in rows:
            r[field] = str(30 * (1 + rng.randrange(30)))
        spec.update(row_id=rows[ordinal % len(rows)]['id'], from_unit='cm', to_unit='m')
    elif kind == 'rule_version':
        first='nth_smallest' if rng.randrange(2) else 'nth_largest'
        rules={'v1':first,'v2':'nth_largest' if first=='nth_smallest' else 'nth_smallest'}
        active='v1' if rng.randrange(2) else 'v2'
        spec.update(operation=rules[active], k=2, skill_id='priority_' + root_id[:6],
                    effective_version=active,rule_definitions=rules)
    elif kind == 'evidence_state':
        spec.update(state=['present', 'retrievable', 'missing_rule', 'conflicting_rules', 'unavailable'][ordinal % 5],
                    answer='ZONE-' + root_id[:6], document_id='doc_' + root_id[:10])
    elif kind == 'operation_state':
        spec.update(status=['submitted', 'running', 'partial', 'failed', 'cancelled', 'completed'][ordinal % 6],
                    operation_id='job_' + root_id[:10])
    changed = copy.deepcopy(spec)
    if kind == 'comparison':
        changed['operation'] = {'gt':'ge', 'lt':'le'}[spec['operation']]
    elif kind == 'ordering':
        changed['operation'] = 'nth_largest'
    elif kind == 'binding':
        if spec['field'] == 'source_id':
            changed['field'] = 'destination_id'
        else:
            changed['row_id'] = rows[(ordinal + 1) % len(rows)]['id']
    elif kind == 'units':
        changed['to_unit'] = 'mm'
    elif kind == 'rule_version':
        active='v2' if spec['effective_version']=='v1' else 'v1'
        changed.update(operation=spec['rule_definitions'][active], effective_version=active)
    elif kind == 'evidence_state':
        changed['state'] = 'missing_rule' if spec['state'] in {'present', 'retrievable'} else 'present'
    else:
        changed['status'] = 'running' if spec['status'] == 'completed' else 'completed'
    invariant = copy.deepcopy(spec)
    invariant['rows'] = list(reversed(invariant['rows']))
    invariant['memory_placement'] = 'back' if spec['memory_placement']=='front' else 'front'
    proof = require_discriminating(spec, changed, invariant)
    variants = {'original': spec, 'counterfactual': changed, 'invariance': invariant}
    return {'root_task_id': root_id, 'split': split, 'ordinal': ordinal, 'semantic_family_id': kind,
        'source_group_id': 'procedural-case-' + root_id, 'variants': variants, 'pair_proof': proof,
        'origin': 'authored_executable_procedural_case', 'independent_natural_source': False,
        'private_canary': 'PRIVATE-ORACLE-' + digest(['canary', root_id])}


def wording_contract(root):
    """Author a fresh question frame. Slots preserve exact references/conditions."""
    kind = root['semantic_family_id']
    requests = {
        'comparison': ('{table} 的 {field} {condition} {threshold}，傳回符合條件的項目 ID。', ['table','field','condition','threshold']),
        'ordering': ('從 {table} 依 {field} 選取 {condition} {rank} 個項目的 ID；同值按 ID 字典順序，順位計算項目而非相異值。', ['table','field','condition','rank']),
        'binding': ('請逐字取出 {table} 中 ID 為 {row} 的 {field} 欄位，保留原本 JSON 型別。', ['table','row','field']),
        'units': ('將 {table} 中 {row} 的 {field} 數值由 {from_unit} 換算成 {to_unit}。', ['table','row','field','from_unit','to_unit']),
        'rule_version': ('依目前明確生效的 {skill} 規則，從 {table} 選出所要求的項目 ID。', ['skill','table']),
        'evidence_state': ('只依所附資料與工具可查文件，確認本案件的配送區域；必要時補查，缺少規則或來源衝突則說明需要確認什麼。', []),
        'operation_state': ('請查詢作業 {operation} 的目前狀態，說明是否已完成，不要把提交或執行中當成完成。', ['operation'])}
    example, slots = requests[kind]
    english = root['ordinal'] % 5 == 0
    if english:
        example = {
            'comparison':'Return the item IDs in {table} whose {field} is {condition} {threshold}.',
            'ordering':'From {table}, select the item at position {rank} when {field} is ordered {condition}. Break ties by ascending ID; count rows, not distinct values.',
            'binding':'Copy the {field} value of row {row} from {table}, preserving its JSON type exactly.',
            'units':'Convert the {field} value of row {row} in {table} from {from_unit} to {to_unit}.',
            'rule_version':'Use the explicitly effective version of {skill} to select the requested item IDs in {table}.',
            'evidence_state':'Using only the supplied records and retrievable documents, identify this case’s delivery region. Look up available evidence if needed, and identify missing rules or unresolved conflicting sources.',
            'operation_state':'Check the current status of operation {operation} and say whether it is complete. Submission or ongoing execution does not imply completion.'}[kind]
    return {'id':root['root_task_id'], 'kind':kind, 'required_slots':slots, 'reference_request':example,
        'style': ['工作同事的簡短請求','自然的客服問題','研究筆記中的資料處理需求','專案狀態查核','簡潔日常用語'][root['ordinal'] % 5],
        'language': 'English' if english else 'Traditional Chinese for Taiwan',
        'variation_seed':root['root_task_id'][:8],
        'constraint': 'Use each required {slot} exactly once. Preserve requested semantics and literal field types. Do not invent facts, answers, operations, deadlines, additional filters, or claims of real side effects. This is a fictional data exercise.'}


def instantiate_question(frame, spec):
    values = {'table':spec['table_id'], 'field':spec['field'],
        'condition':PHRASES.get(spec.get('operation'), ''), 'threshold':spec.get('threshold',''),
        'rank':str(spec.get('k','')), 'row':spec.get('row_id',''),
        'from_unit':spec.get('from_unit',''), 'to_unit':spec.get('to_unit',''),
        'skill':spec.get('skill_id',''), 'operation':spec.get('operation_id','')}
    if not any('\u4e00' <= ch <= '\u9fff' for ch in frame):
        values['condition'] = {'gt':'strictly greater than','ge':'greater than or equal to',
            'lt':'strictly less than','le':'less than or equal to','nth_smallest':'ascending',
            'nth_largest':'descending'}.get(spec.get('operation'),'')
    return frame.format(**values)


def public_world(spec, frame):
    """Public context, tool registry and executor world contain no answer keys."""
    kind = spec['kind']
    tables = {spec['table_id']: copy.deepcopy(spec['rows'])}
    context, skills, evidence, ledger = [], [], {}, StateLedger()
    names = []
    if kind in {'comparison','ordering','binding','units','rule_version'}:
        context.append({'id':spec['table_id'], 'title':'Fictional project table', 'revision':'1',
                        'text':canonical({'rows':spec['rows'], 'numeric_columns_use_decimal_strings':True})})
    if kind in {'comparison','ordering','rule_version'}:
        names = ['table_select']
        skills.append(library_card('comparison-by-dimensions' if kind=='comparison' else 'ordered-top-k'))
        if kind == 'rule_version':
            sid = spec['skill_id']
            for version, op in spec['rule_definitions'].items():
                skills.append({'id':sid,'version':version,'text':f'Fictional procedure: use {op} with k=2 on column {spec["field"]}. Ties: ID ascending. This definition applies only when this version is declared effective.'})
            context.append({'id':'active-rule','title':'Effective version declaration','revision':'1',
                            'text':f'The currently effective version of {sid} is {spec["effective_version"]}. Other versions are inactive.'})
    elif kind == 'binding':
        names = ['resolve_value']
        skills.append(library_card('literal-typed-tool-arguments'))
    elif kind == 'units':
        names = ['convert_unit']
        skills.append(library_card('unit-conversion'))
    elif kind == 'evidence_state':
        state, did = spec['state'], spec['document_id']
        # The public document is a fact source, never a private target annotation.
        document = {'id':did, 'title':'Case delivery record', 'revision':'1',
                    'text':'This fictional case is assigned to delivery region ' + spec['answer'] + '.'}
        if state == 'present':
            context.append(document)
        elif state == 'retrievable':
            evidence[did] = document
            context.append({'id':'catalog','title':'Source catalog','revision':'1','text':f'The delivery record is in document {did}. Retrieve it with read_evidence.'})
            names = ['read_evidence', 'search_evidence']
        elif state == 'missing_rule':
            context.append({'id':'case','title':'Case facts','revision':'1','text':'The case has region code X2. The rule mapping region codes to delivery regions has not been provided. No tool can retrieve the mapping.'})
        elif state == 'conflicting_rules':
            context.extend([document, {'id':did+'-other','title':'Conflicting record','revision':'1','text':'This same fictional case is assigned to delivery region ZONE-CONFLICT. Both records have equal authority and no precedence rule is available.'}])
        else:
            context.append({'id':'catalog','title':'Source catalog','revision':'1','text':f'Delivery information should be in document {did}. It is not in the visible context. You may attempt read_evidence.'})
            names = ['read_evidence']
        skills.extend([library_card('grounded-evidence-answer'), library_card('clarify-missing-inputs')])
    else:
        op = spec['operation_id']
        ledger.submit(op, {'fictional_task':'process case'})
        if spec['status'] != 'submitted':
            ledger.event(op, 'running')
        if spec['status'] not in {'submitted','running'}:
            ledger.event(op, spec['status'], {'observed':'simulator event'})
        names = ['read_operation_state']
        context.append({'id':'operation-ledger','title':'Local simulator','revision':str(ledger.revision),
                       'text':f'This is a versioned local simulator, not an external job. Operation ID: {op}. Query its current observable status.'})
        skills.append(library_card('external-state-grounding'))
    public = {'task':instantiate_question(frame,spec), 'context':context, 'skills':skills,
              'tools':[tool_schema(n) for n in names], 'memory_placement':spec['memory_placement']}
    runtime = Executor(tables=tables, evidence=evidence, skills={s['id']+'@'+s['version']:s for s in skills}, ledger=ledger)
    return public, runtime


def build_trajectory(root, variant, frame):
    spec = root['variants'][variant]
    public, runtime = public_world(spec, frame)
    messages = actor_messages(public, forbidden_canaries=[root['private_canary']])
    kind = spec['kind']
    english = not any('\u4e00' <= ch <= '\u9fff' for ch in frame)
    def call(name, args):
        cid = 'call_' + str(len(runtime.receipts))
        messages.append({'role':'assistant','content':'','loss':True,
                         'tool_calls':[{'id':cid,'type':'function','function':{'name':name,'arguments':args}}]})
        result = runtime.call(name,args)
        messages.append({'role':'tool','name':name,'tool_call_id':cid,'content':canonical(result),'loss':False})
        return result
    if kind in {'comparison','ordering','rule_version'}:
        args = {k:spec[k] for k in ['table_id','field','operation']}
        args.update({k:spec[k] for k in ['threshold','k'] if k in spec})
        args['tie_rule'] = 'id_ascending'
        result = call('table_select',args)
        if not verify_execution(spec,result):
            raise ValueError('Independent selection oracle rejected execution')
        answer = '符合條件的項目 ID：' + ('、'.join(result['result']['ids']) or '沒有符合條件的項目') + '。'
        if english:
            answer = 'Matching item IDs: ' + (', '.join(result['result']['ids']) or 'none') + '.'
    elif kind in {'binding','units'}:
        ref = {k:spec[k] for k in ['table_id','row_id','field']}
        if kind == 'binding':
            result = call('resolve_value',ref)
            answer = canonical(result['result']['value'])
        else:
            result = call('convert_unit',{'value_ref':ref,'from_unit':spec['from_unit'],'to_unit':spec['to_unit']})
            answer = result['result']['value'] + ' ' + result['result']['unit']
        if not verify_execution(spec,result):
            raise ValueError('Independent typed-value oracle rejected execution')
    elif kind == 'evidence_state':
        state = spec['state']
        if state == 'retrievable':
            result = call('read_evidence',{'document_id':spec['document_id']})
            if not result['ok'] or spec['answer'] not in result['result']['text']:
                raise ValueError('Evidence result does not support answer')
        elif state == 'unavailable':
            result = call('read_evidence',{'document_id':spec['document_id']})
            if result['ok']:
                raise ValueError('Unavailable source unexpectedly available')
        if state in {'present','retrievable'}:
            answer = f'依文件 {spec["document_id"]}，配送區域是 {spec["answer"]}。'
        elif state == 'missing_rule':
            answer = '目前缺少區域代碼與配送區域的對照規則。請提供代碼 X2 對應的規則，才能確定配送區域。'
        elif state == 'conflicting_rules':
            answer = '兩份同等來源對同一案件的配送區域記載互相矛盾，且沒有優先規則。請確認應採用哪份來源；目前無法確定配送區域。'
        else:
            answer = '工具未能取得所需的配送文件，因此目前無法確認配送區域。請提供該文件或可查證的替代來源。'
        if english:
            if state in {'present','retrievable'}:
                answer = f'According to document {spec["document_id"]}, the delivery region is {spec["answer"]}.'
            elif state == 'missing_rule':
                answer = 'The mapping from region code X2 to a delivery region is missing. Please provide that rule so I can identify the region.'
            elif state == 'conflicting_rules':
                answer = 'The two equally authoritative records conflict, and no precedence rule is given. Please clarify which source applies; I cannot determine the region from these records.'
            else:
                answer = 'The tool could not retrieve the required delivery document, so the region remains unconfirmed. Please provide the document or another verifiable source.'
    else:
        result = call('read_operation_state', {'operation_id':spec['operation_id']})
        if not verify_execution(spec,result):
            raise ValueError('State observation differs from independent success predicate')
        answer = f'作業目前狀態為 {result["result"]["status"]}。' + ('已完成。' if spec['status']=='completed' else '尚不能宣稱已完成。')
        if english:
            answer = f'The operation status is {result["result"]["status"]}. ' + ('It is complete.' if spec['status']=='completed' else 'It cannot be reported as complete.')
    messages.append({'role':'assistant','content':answer,'loss':True})
    return {'public':public,'messages':messages,'execution_receipts':runtime.receipts,
            'final_answer':answer, 'oracle_checked':True}
