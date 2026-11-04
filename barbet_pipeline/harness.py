"""Public-only bounded action loop shared by live inference and recordings.

No oracle import, task-specific repair or hidden success predicate is available.
An answer event is an observed model proposal, not a declaration of task success.
"""
from __future__ import annotations
import copy
from dataclasses import asdict

from .common import canonical
from .visibility import actor_messages
from barbet_inference.backends import Generation
from barbet_inference.context_agent import parse_action
from barbet_inference.sft import pangolin_render
from testtime.native_calls import extract_final


def run_public_episode(backend, public, executor, *, max_steps=6, output_tokens=512):
    if type(max_steps) is not int or not 1<=max_steps<=32 or type(output_tokens) is not int or output_tokens<1:
        raise ValueError('Invalid public action budget')
    messages=actor_messages(public)
    declared={t['function']['name'] for t in public['tools']}
    if len(declared)!=len(public['tools']):raise ValueError('Duplicate declared tools')
    trace=[];seen=set()
    for step in range(max_steps):
        text,_=pangolin_render(messages,public['tools'])
        ids=backend.tokenizer.encode(text+'<|assistant_channel|>',add_special_tokens=False)
        if hasattr(ids,'ids'):ids=ids.ids
        if len(ids)+output_tokens>backend.max_model_len:
            return {'status':'context_budget_exhausted','trace':trace,'messages':messages}
        completion=backend.generate([Generation(ids,{'temperature':0,'max_tokens':output_tokens})])[0]
        event={'step':step,'raw_proposal':asdict(completion),'host_decision':None,'effects':[]}
        trace.append(event)
        if completion.finish_reason!='stop':
            event['host_decision']='incomplete_generation'
            return {'status':'incomplete_generation','trace':trace,'messages':messages}
        content,unclosed,_=extract_final(completion.text,False,terminal_strings=backend.terminal_strings)
        if unclosed or not content:
            event['host_decision']='invalid_generation'
            return {'status':'invalid_generation','trace':trace,'messages':messages}
        try:
            prose,calls=parse_action(content)
        except ValueError:
            event['host_decision']='invalid_tool_encoding'
            return {'status':'invalid_tool_encoding','trace':trace,'messages':messages}
        if not calls:
            event['host_decision']='answer_proposal_recorded_not_semantically_graded'
            messages.append({'role':'assistant','content':content,'loss':False})
            return {'status':'answer_proposed','answer':content,'trace':trace,'messages':messages,
                    'semantic_success':'NOT_GRADED_BY_HOST'}
        normalized=[]
        for call in calls:
            fn=call.get('function',{}) if isinstance(call,dict) else {}
            cid=call.get('id') if isinstance(call,dict) else None
            if (not isinstance(cid,str) or not cid or cid in seen or fn.get('name') not in declared
                    or not isinstance(fn.get('arguments'),dict)):
                event['host_decision']='undeclared_or_invalid_call'
                return {'status':'call_rejected','trace':trace,'messages':messages}
            seen.add(cid)
            normalized.append({'id':cid,'type':'function','function':copy.deepcopy(fn)})
        event['host_decision']='dispatch_without_semantic_repair'
        messages.append({'role':'assistant','content':prose,'tool_calls':normalized,'loss':False})
        for c in normalized:
            f=c['function'];out=executor.call(f['name'],f['arguments'])
            event['effects'].append(copy.deepcopy(executor.receipts[-1]))
            messages.append({'role':'tool','name':f['name'],'tool_call_id':c['id'],
                             'content':canonical(out),'loss':False})
    return {'status':'action_budget_exhausted','trace':trace,'messages':messages}
