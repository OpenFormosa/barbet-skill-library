"""Answer-free inference requests for the versioned multi-turn SFT records."""
import hashlib
import json
from .sft import inference_messages, portable_messages


def prepare_dataset_request(row, *, model, mode='chat', target_turn=None, max_tokens=4096):
    if type(max_tokens) is not int or max_tokens<1:
        raise ValueError('max_tokens must be positive')
    messages=json.loads(row['messages_json'])
    tools=json.loads(row['tools_json'])
    if target_turn is None:
        target_turn=max(i for i,m in enumerate(messages) if m['role']=='assistant' and m.get('loss'))
    if not 0<=target_turn<len(messages) or messages[target_turn]['role']!='assistant' or not messages[target_turn].get('loss'):
        raise ValueError('target must be a supervised assistant turn')
    if mode=='chat':
        history=inference_messages(messages,target_turn)
        native=portable_messages(history,tools)
        return {'model':model,'messages':[{k:v for k,v in m.items() if k!='loss'} for m in native],
                'max_tokens':max_tokens,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
    if mode not in {'barbet_completion','pangolin_completion','minicpm_completion'}:
        raise ValueError('unknown request mode')
    mini=mode=='minicpm_completion'
    spans=json.loads(row['mini_target_spans_json' if mini else 'target_spans_json'])
    span=next((s for s in spans if s[2]==target_turn),None)
    if span is None:
        raise ValueError('target span missing')
    text=row['mini_text' if mini else 'text'][:span[0]]
    # Stored IDs belong to their named tokenizer. Text is supplied once here;
    # the completion server must not apply another chat template or add BOS.
    return {'model':model,'prompt':text,'max_tokens':max_tokens,'temperature':0,
            'add_special_tokens':False}
