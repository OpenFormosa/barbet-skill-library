"""Memory + skills + typed tools + bounded, feedback-driven refinement.

The application supplies real tool/verifier callbacks. This module never runs a
shell, imports a retrieved program, or executes a tool mentioned only in text.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json

from .backends import Generation
from .library import SkillLibrary
from .memory import MemoryItem, MemoryStore, Verification, compose_memory_prompt, canonical
from .sft import SYSTEM, RESERVED, portable_messages, prepare_tools
from .tool_contract import undeclared_arguments
from testtime.native_calls import extract_final, strict_json


def parse_action(text):
    """Read a terminal JSON tool envelope, preserving optional preceding prose."""
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, end = json.JSONDecoder().raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and 'tool_calls' in value and not text[index+end:].strip():
            value = strict_json(text[index:index+end])
            if set(value) != {'tool_calls'} or not isinstance(value['tool_calls'], list) or not value['tool_calls']:
                raise ValueError('invalid tool action envelope')
            return text[:index].strip(), value['tool_calls']
    if '"tool_calls"' in text:
        raise ValueError('malformed terminal tool action')
    return text, []


class ContextAgent:
    def __init__(self, backend, *, library=None, skills=(), tools=(), handlers=None,
                 memory=(), verifier=None, output_tokens=4096, max_steps=8,
                 max_refinements=2, top_k=4, memory_placement='back'):
        if not 1 <= max_steps <= 64 or not 0 <= max_refinements <= 2:
            raise ValueError('invalid iteration budget')
        if not isinstance(output_tokens, int) or output_tokens < 1:
            raise ValueError('invalid output budget')
        if memory_placement not in {'front','back'}:
            raise ValueError('invalid memory placement')
        self.backend, self.library = backend, library or SkillLibrary()
        self.skills = list(skills)
        self.tools, self.validators = prepare_tools(list(tools))
        self.handlers, self.verifier = handlers or {}, verifier
        if set(self.handlers) - set(self.validators):
            raise ValueError('handler without declared tool schema')
        self.memory = MemoryStore(memory)
        self.memory_turn = 0
        self.output_tokens, self.max_steps = output_tokens, max_steps
        self.max_refinements, self.top_k = max_refinements, top_k
        self.memory_placement = memory_placement

    def _tokens(self, messages):
        native = [{k:v for k,v in m.items() if k != 'loss'} for m in portable_messages(messages,self.tools)]
        if any(RESERVED.search(m['content']) for m in native):
            raise ValueError('reserved native delimiter in live content; supply a quoted safe representation')
        ids = self.backend.tokenizer.apply_chat_template(native, tools=[], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, 'get'):
            ids = ids['input_ids']
        if ids and isinstance(ids[0],list):
            raise ValueError('unexpected batched tokenizer response')
        if len(ids)+self.output_tokens>self.backend.max_model_len:
            raise ValueError('context budget exceeded; no history was truncated')
        return list(ids)

    def run(self, question):
        if not isinstance(question,str) or not question.strip():
            raise ValueError('question is required')
        skill_ids = self.skills or (['tool-selection-and-arguments','external-state-grounding']
                                   if self.tools else ['grounded-evidence-answer'])
        skill_text, skill_refs = self.library.render(skill_ids)
        base_turn = self.memory_turn
        selected = self.memory.retrieve(question,turn=base_turn,top_k=self.top_k)
        messages = [{'role':'system','content':SYSTEM+'\n\nAPPLICABLE SKILLS:\n'+skill_text},
                    {'role':'user','content':compose_memory_prompt(question,selected,turn=base_turn,placement=self.memory_placement)}]
        trace, seen_candidates, seen_ids = [], set(), set()
        refinements = 0
        def result(status, **extra):
            self.memory_turn = max(self.memory_turn,base_turn+len(messages))
            return {'status':status,'trace':trace,'messages':messages,'skills':skill_refs,
                    'memory_ids':[m.id for m in selected],**extra}
        for step in range(self.max_steps):
            ids=self._tokens(messages)
            completions=self.backend.generate([Generation(ids,{'max_tokens':self.output_tokens,'temperature':0.0})])
            if len(completions)!=1:
                raise ValueError('expected one completion')
            completion=completions[0]
            trace.append({'step':step,'prompt_tokens':completion.prompt_tokens,
                          'output_tokens':completion.output_tokens,'finish_reason':completion.finish_reason})
            if completion.finish_reason!='stop':
                return result('generation_incomplete')
            text,unclosed,_=extract_final(completion.text,False,terminal_strings=self.backend.terminal_strings)
            if unclosed or not text:
                return result('invalid_generation')
            try:
                prose,calls=parse_action(text)
            except ValueError as exc:
                return result('invalid_tool_action',error=str(exc))
            if calls:
                normalized=[]
                for n,call in enumerate(calls):
                    if not isinstance(call,dict):
                        return result('invalid_tool_action')
                    fn=call.get('function',call)
                    if not isinstance(fn,dict):
                        return result('invalid_tool_action')
                    name,args=fn.get('name'),fn.get('arguments')
                    if name not in self.validators or not isinstance(args,dict):
                        return result('invalid_tool_action')
                    if next(self.validators[name].iter_errors(args),None) is not None:
                        return result('invalid_tool_arguments')
                    if undeclared_arguments(args,self.validators[name].schema):
                        return result('unconfirmed_tool_arguments')
                    cid=call.get('id',f'live_{step}_{n}')
                    if not isinstance(cid,str) or not cid or cid in seen_ids:
                        return result('invalid_call_identity')
                    seen_ids.add(cid)
                    normalized.append({'id':cid,'type':'function','function':{'name':name,'arguments':args}})
                if RESERVED.search(canonical(normalized)):
                    return result('unsupported_native_literal')
                messages.append({'role':'assistant','content':prose,'tool_calls':normalized})
                if any(c['function']['name'] not in self.handlers for c in normalized):
                    return result('awaiting_tool_execution',tool_calls=normalized)
                # Only pre-registered callbacks run; all arguments were validated
                # before the first side effect in this batch.
                for call in normalized:
                    fn=call['function']
                    try:
                        observed=self.handlers[fn['name']](**fn['arguments'])
                        content=observed if isinstance(observed,str) else canonical(observed)
                    except Exception as exc:
                        # Do not expose an arbitrary exception message containing
                        # credentials; the callback may return a deliberate public
                        # error object when detailed diagnostics are appropriate.
                        content=canonical({'error':type(exc).__name__,'completion_status':'unknown'})
                    messages.append({'role':'tool','name':fn['name'],'tool_call_id':call['id'],'content':content})
                    self.memory_turn=base_turn+len(messages)-1
                    remembered=canonical({'call':fn,'observation':content})
                    self.memory.add(MemoryItem(f'observation:{base_turn}:'+call['id'],remembered,
                        'registered_callback:'+fn['name'],self.memory_turn,'tool_observation','actual_callback_result'))
                continue
            candidate_sha=hashlib.sha256(text.encode()).hexdigest()
            if candidate_sha in seen_candidates:
                return result('refinement_stalled',answer=text)
            seen_candidates.add(candidate_sha)
            messages.append({'role':'assistant','content':text})
            if self.verifier is None:
                return result('answered_without_external_verification',answer=text)
            check=self.verifier(text)
            if not isinstance(check,Verification) or check.candidate_sha256!=candidate_sha:
                raise ValueError('stale or unbound verifier feedback')
            if type(check.passed) is not bool or not check.scope or not check.feedback:
                raise ValueError('invalid verifier result')
            trace[-1]['verification']=asdict(check)
            if check.passed:
                return result('verified_within_scope',answer=text,verification_scope=check.scope)
            if refinements>=self.max_refinements:
                return result('refinement_budget_exhausted',answer=text)
            refinements+=1
            messages.append({'role':'user','content':'Actual verifier feedback:\n'+canonical(asdict(check))+\
                             '\nRevise the answer using this feedback and preserve the original task.'})
        return result('step_budget_exhausted')
