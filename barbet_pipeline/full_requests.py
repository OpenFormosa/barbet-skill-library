"""Answer-free inference adapter bound to the full-v4 measured prefix hashes."""
import json
from .common import digest
from barbet_inference.sft import portable_messages,pangolin_render


def prepare_full_request(row,*,model,mode='chat',target_turn=None):
    if row['schema_version']!='barbet-full-reprocess-v4':raise ValueError('Unsupported full-data schema')
    messages=json.loads(row['messages_json']);tools=json.loads(row['tools_json'])
    proofs=json.loads(row['inference_decisions_json'])
    selected=next((p for p in proofs if target_turn is None or p['target_turn']==target_turn),None)
    if selected is None:raise ValueError('No measured accepted assistant decision at this turn')
    turn=selected['target_turn']
    if messages[turn]['role']!='assistant' or not messages[turn]['loss']:raise ValueError('Target is not accepted assistant content')
    prefix=messages[:turn]
    if mode=='chat':
        visible=portable_messages(prefix,tools)
        for message in visible:
            message.pop('loss',None)
            if message['role']=='tool':
                message['role']='user';message['content']='TOOL_OBSERVATION:\n'+message['content']
        if digest(visible)!=selected['chat_sha256']:raise ValueError('Saved Chat prefix identity changed')
        return {'model':model,'messages':visible,'max_tokens':selected['output_reserve_tokens'],
                'chat_template_kwargs':{'enable_thinking':False}}
    if mode=='completion':
        prompt,_=pangolin_render(prefix,tools);prompt+='<|assistant_channel|>'
        if digest(prompt)!=selected['completion_sha256']:raise ValueError('Saved Completion prefix identity changed')
        return {'model':model,'prompt':prompt,'add_special_tokens':False,'max_tokens':selected['output_reserve_tokens']}
    raise ValueError('Mode must be chat or completion')
