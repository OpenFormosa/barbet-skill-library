"""Real, short native-backend smoke. Never a task-capability benchmark."""
import json
import sys
from pathlib import Path
import torch
torch.set_num_threads(4)
# Existing model runtime is preserved; jsonschema is an isolated data dependency.
sys.path.append('/usr/local/lib/python3.12/dist-packages')
from barbet_pipeline.common import read_json, save_new
from barbet_inference.barbet_backend import NativeBarbetBackend
from barbet_inference.backends import Generation
from barbet_inference.sft import pangolin_render


def run(folder, manifest, output):
    backend = NativeBarbetBackend(folder, read_json(manifest), max_model_len=8192)
    messages = [{'role':'user','content':'請簡短說明資料不足時應該怎麼做。'}]
    ids = backend.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    completion = backend.generate([Generation(ids, {'temperature':0, 'max_tokens':12})])[0]
    # Exercise incremental decode and a separately computed full-prefix top-1.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    direct_ids, cached_ids, differences = [], [], []
    cache = None
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for step in range(3):
            full = torch.tensor([ids + cached_ids], device='cuda:0')
            cached_input = full if step == 0 else full[:, -1:]
            cached = backend.model.model(input_ids=cached_input, past_key_values=cache, use_cache=True, return_dict=True)
            cache = cached.past_key_values
            fast = backend.model.lm_head(cached.last_hidden_state[:, -1, :]).float()
            slow = backend.model.lm_head(backend.model.model(input_ids=full, use_cache=False, return_dict=True).last_hidden_state[:, -1, :]).float()
            assert torch.isfinite(fast).all() and torch.isfinite(slow).all()
            cached_ids.append(int(fast.argmax())); direct_ids.append(int(slow.argmax()))
            differences.append(float((fast - slow).abs().max()))
    tools = [{'type':'function','function':{'name':'read_evidence','parameters':{'type':'object','properties':{'id':{'type':'string'}},'required':['id'],'additionalProperties':False}}}]
    resumed = [{'role':'system','content':'Use the registered tools and report observed status.'},
        {'role':'user','content':'讀取文件 d1，回答狀態。'},
        {'role':'assistant','content':'','tool_calls':[{'id':'r1','type':'function','function':{'name':'read_evidence','arguments':{'id':'d1'}}}]},
        {'role':'tool','tool_call_id':'r1','name':'read_evidence','content':'{"id":"d1","status":"running"}'}]
    text, _ = pangolin_render(resumed, tools)
    continuation = backend.generate([Generation(backend.tokenizer.encode(text+'<|assistant_channel|>'), {'temperature':0,'max_tokens':12})])[0]
    result = {'backend':backend.metadata,'short_generation':vars(completion),
        'portable_tool_history_continuation':vars(continuation),
        'cached_vs_full_prefix':{'positions':3,'same_argmax':cached_ids == direct_ids,'max_abs_logit_differences':differences},
        'precision_census_enforced_by_release_loader':True,
        'capability_score':None,'task_semantic_success':'NOT_EVALUATED',
        'training_parity':'NOT_RUN','native1M_generation':'NOT_RUN'}
    save_new(output,result)
    print(json.dumps(result,ensure_ascii=False))

if __name__ == '__main__':
    run(*sys.argv[1:])
