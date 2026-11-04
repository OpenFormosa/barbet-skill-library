"""Actual live token-ID controls for the full 131072-budget export protocol."""
import json,sys,time
from pathlib import Path
from urllib import request
from transformers import AutoTokenizer
sys.path.append('/root/barbet-context-sft-data/.venv/lib/python3.12/site-packages')
from barbet_pipeline.common import save_new,digest,sha_file
from barbet_pipeline.full_corpus import QWEN_PATH,STATE,MAX_TOTAL

tok=AutoTokenizer.from_pretrained(str(QWEN_PATH),trust_remote_code=False,local_files_only=True)
fixtures=[
 [{'role':'system','content':'Use the supplied data.'},{'role':'user','content':'甲＝6；乙＝7。找出嚴格大於6的項目。'}],
 [{'role':'system','content':'Use explicit tools.'},{'role':'user','content':'Find the status.'},{'role':'assistant','content':'{"tool_calls":[]}'},{'role':'user','content':'TOOL_OBSERVATION:\n{"status":"cancelled","done":false}'}]
]
results=[]
for messages in fixtures:
 for mode in ['chat','completion']:
  if mode=='chat':
   text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   body={'messages':messages,'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}}
  else:
   text='<s><|system|>Use supplied data.\n<|user_channel|>'+messages[-1]['content']+'\n<|assistant_channel|>'
   body={'prompt':text}
  ids=tok.encode(text,add_special_tokens=False);body.update(model='Qwen/Qwen3.8-Flash-Next-FP8',add_special_tokens=False)
  req=request.Request('http://127.0.0.1:8002/tokenize',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with request.urlopen(req,timeout=60) as response:r=json.load(response)
  assert r['tokens']==ids and r['count']==len(ids) and r['max_model_len']==MAX_TOTAL
  results.append({'mode':mode,'ids_sha256':digest(ids),'exact_ids_equal':True,'count':len(ids)})
save_new(STATE/'live-tokenizer-controls.json',{'passed':True,'budget':MAX_TOTAL,'output_reserve':4096,
 'tokenizer_sha256':sha_file(QWEN_PATH/'tokenizer.json'),'tokenizer_config_sha256':sha_file(QWEN_PATH/'tokenizer_config.json'),
 'controls':results,'scope':'tokenizer/template/configured budget only; no long generation or Barbet vLLM certification'})
print(json.dumps({'passed':True,'controls':len(results),'budget':MAX_TOTAL}))
