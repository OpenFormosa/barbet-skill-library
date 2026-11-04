"""Offline oracle conditions. Never imported by the public harness."""
from __future__ import annotations
import copy
from .common import canonical
from .factory import public_world, build_trajectory
from .visibility import actor_messages


def oracle_conditions(root,variant,frame):
    spec=root['variants'][variant]
    public,_=public_world(spec,frame)
    trajectory=build_trajectory(root,variant,frame)
    base=actor_messages(public)
    states={}
    for level in range(6):
        condition='O'+str(level)
        visible=copy.deepcopy(base)
        addition=None
        if level==1:
            addition={'correct_documents':public['context'],
                      'scope':'Already-visible documents in this atomic pilot; not a retrieval benchmark'}
        elif level==2:
            addition={'sufficient_public_documents':public['context'],'rules':public['skills'],
                      'scope':'Task may be intentionally underdetermined; no missing world fact is invented'}
        elif level==3:
            addition={'binding':{k:spec[k] for k in ['table_id','field','row_id','from_unit','to_unit','document_id','operation_id'] if k in spec}}
        elif level==4:
            addition={'correct_actions':[m['tool_calls'] for m in trajectory['messages'] if m.get('tool_calls')],
                      'information_state':spec.get('state')}
        elif level==5:
            addition={'actual_execution_results':[r['output'] for r in trajectory['execution_receipts']]}
        if addition is not None:
            visible.append({'role':'user','content':'OFFLINE ORACLE DIAGNOSTIC '+condition+': '+canonical(addition),'loss':False})
        states[condition]={'condition':condition,'messages':visible,'tools':copy.deepcopy(public['tools']),
                           'oracle_assisted':level>0,'eligible_for_main_score':level==0,
                           'model_results':'NOT_RUN'}
    return states
