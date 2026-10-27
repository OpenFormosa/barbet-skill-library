"""Prospective, source-specific syntax adapters. No execution or answer generation."""
from collections import Counter
import json
import re
from barbet_inference.sft import Rejected, clean_text
from barbet_inference.memory import canonical

QWEN_HEAD='# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n'
QWEN_TAIL='\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'


def decoded(value):
    if isinstance(value,str):
        try:return json.loads(value)
        except ValueError as exc:raise Rejected('source_json_invalid') from exc
    return value


def exact_tool_boilerplate(text,tools):
    start,end='<|im_system|>tool_declare<|im_middle|>','<|im_end|>'
    try:
        if text.startswith(start) and text.endswith(end):
            return canonical(json.loads(text[len(start):-len(end)]))==canonical(tools)
        if text.startswith(QWEN_HEAD) and text.endswith(QWEN_TAIL):
            body=text[len(QWEN_HEAD):-len(QWEN_TAIL)]
            return canonical([json.loads(s) for s in body.splitlines() if s.strip()])==canonical(tools)
    except ValueError:return False
    return False


def toucan(row):
    quality=decoded(row.get('response_quality_assessment')) or {}
    qquality=decoded(row.get('question_quality_assessment')) or {}
    def score(obj,key):
        value=obj.get(key,{})
        return value.get('score',-1) if isinstance(value,dict) else -1
    if (score(quality,'completeness')!=5 or quality.get('overall_score',-1)<4
            or score(quality,'conciseness')<3 or quality.get('order_correctness') is not True
            or quality.get('desired_tools_used_percentage')!=1
            or score(qquality,'question_quality')<4):
        raise Rejected('toucan_original_rubric_not_complete')
    tools=decoded(row.get('available_tools'))
    if not isinstance(tools,list) or not tools:raise Rejected('toucan_missing_tools')
    raw=decoded(row.get('messages'))
    if not isinstance(raw,list):raise Rejected('toucan_missing_messages')
    out=[];pending={}
    for n,m in enumerate(raw):
        if not isinstance(m,dict):raise Rejected('toucan_invalid_message')
        role=m.get('role');content=m.get('content') or ''
        if role=='system' and isinstance(content,str) and exact_tool_boilerplate(content,tools):continue
        if role=='function':
            name=m.get('name');matches=[cid for cid,fn in pending.items() if fn==name]
            if len(matches)!=1:raise Rejected('toucan_ambiguous_function_response')
            cid=matches[0];del pending[cid]
            out.append({'role':'tool','name':name,'tool_call_id':cid,'content':content});continue
        if role not in {'system','user','assistant'}:raise Rejected('toucan_unknown_source_role')
        item={'role':role,'content':content}
        if role=='assistant' and m.get('function_call'):
            call=decoded(m['function_call'])
            if not isinstance(call,dict):raise Rejected('toucan_invalid_function_call')
            name=call.get('name');cid=f'toucan_{n}'
            item['tool_calls']=[{'id':cid,'type':'function','function':call}]
            pending[cid]=name
        # Empty private-reasoning messages contain no public training target.
        if role=='assistant' and not clean_text(content,assistant=True).strip() and not item.get('tool_calls'):continue
        if role=='assistant' and out and out[-1]['role']=='assistant':
            out[-1]['content']+='\n'+content if content else ''
            if item.get('tool_calls'):out[-1].setdefault('tool_calls',[]).extend(item['tool_calls'])
        else:out.append(item)
    query=next((m['content'] for m in out if m['role']=='user'),'')
    meta=decoded(row.get('metadata')) or {}
    servers=sorted(str(m.get('server_id')) for m in meta.get('mcp_servers',[]) if isinstance(m,dict))
    # Toolset is part of identity; prompt indices repeat across server sets.
    root=re.sub(r'(_follow_up|_followup)$','',str(meta.get('prompt_id',row.get('uuid',''))))
    group='toucan-root:'+canonical({'servers':servers,'prompt':root})
    task='toucan-user-turns:'+canonical([m['content'] for m in out if m['role']=='user'])
    return {'query':query,'messages':out,'tools':tools,'group':group,'task_key':task,
        'domain':'tool_use','license':'apache-2.0','upstream_id':str(row.get('uuid','')),
        'verification_tier':'upstream_rubric_complete_not_locally_semantic_verified',
        'observation_origin':'upstream_recorded_mcp_result','selection_signal':'original_rubric_thresholds'}


CONTEXT=re.compile(r'(?is)(definition of a task first|(?:read|given|following|provided|above|below).{0,65}(?:paragraph|passage|article|text|sentence|dialogue|conversation|premise|data|table|review|document|context|information)|(?:premise|hypothesis|paragraph|passage|context|article|input|dialogue|review)\s*:|translat(?:e|ion).{0,100}:|summariz(?:e|ation).{0,100}:)')
LOGIC=re.compile(r'(?is)(how (?:many|much|long|far)|calculate|compute|solve|what is (?:the sum|the difference|the product)|which (?:comes|happens).{0,30}(?:before|after)|sort (?:these|the following)|alphabetical order)')
CODING=re.compile(r'(?is)(write (?:a |the )?(?:python|function|program|code)|def [a-zA-Z_]|```(?:python|javascript|java|c\+\+))')


def openorca(row):
    query=row.get('question','');answer=row.get('response','');sid=row.get('id','')
    if not isinstance(query,str) or not isinstance(answer,str) or not sid:
        raise Rejected('openorca_incomplete_source')
    # Source-only, conservative heuristic. A question or MCQ option list alone
    # is not evidence; never manufacture context from the answer.
    if CONTEXT.search(query) and len(query)>=140:
        domain='reading';signal='explicit_provided_text_or_task_definition'
    elif LOGIC.search(query) and len(re.findall(r'\b\d+(?:\.\d+)?\b',query))>=2:
        domain='math';signal='numerical_premises_and_operation'
    elif CODING.search(query) and len(query)>=180:
        domain='coding';signal='explicit_code_contract'
    else:raise Rejected('openorca_no_supported_context_logic_signal')
    raw=[]
    if row.get('system_prompt'):raw.append({'role':'system','content':row['system_prompt']})
    raw.extend([{'role':'user','content':query},{'role':'assistant','content':answer}])
    return {'query':query,'messages':raw,'tools':[],'group':'openorca-seed:'+sid,
        'task_key':'openorca-seed:'+sid,'domain':domain,'license':'mit','upstream_id':sid,
        'verification_tier':'upstream_public_response_not_locally_semantic_verified',
        'observation_origin':'user_problem_premises','selection_signal':signal}
