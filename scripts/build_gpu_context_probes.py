from pathlib import Path
from collections import Counter,defaultdict
import json,copy
import pyarrow as pa
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json,save_new,sha_file,digest
from barbet_pipeline.visibility import actor_messages
from barbet_pipeline.export import load_tokenizer
from barbet_inference.sft import pangolin_render,tokenize_spans

s=Path('/root/barbet-context-sft-data/reports/full-corpus-v4/gpu-quality');root=Path('/root/barbet-context-sft-data/reports/semantic-curriculum-v3/pilot/records')
tokenizer=load_tokenizer('/root/.cache/huggingface/hub/models--voidful--barbet-skill/snapshots/67131067ff1825179d5e38e91d499159669e9c66/tokenizer.json')
groups=defaultdict(dict);inputs=[]
for path in sorted(root.glob('train-*.parquet')):
 inputs.append({'file':str(path),'sha256':sha_file(path)})
 for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
  for r in batch.to_pylist():groups[r['root_task_id']][r['variant']]=r
assert len(groups)==2000
rows=[];counts=Counter()
def append(root_id,family,condition,prefix,target,tools,correct_candidate,source_id):
 text,spans=pangolin_render(prefix+[target],tools);ids,labels=tokenize_spans(tokenizer,text,spans)
 assert len(spans)==1 and labels[0]==-100 and len(ids)<=8192
 rows.append({'record_id':digest([root_id,condition,correct_candidate,source_id]),'root_task_id':root_id,'semantic_family_id':family,'condition':condition,'correct_candidate':correct_candidate,'source_record_id':source_id,'input_ids':ids,'labels':labels,'pangolin_tokens':len(ids),'pangolin_supervised_tokens':sum(i!=-100 for i in labels)})
for rid,views in sorted(groups.items()):
 assert set(views)=={'original','counterfactual','invariance'}
 originals={};targets={};tools={};texts={}
 for name,row in views.items():
  messages=json.loads(row['messages_json']);span=json.loads(row['target_spans_json'])[0]
  assert span[2]==2 and all(not m.get('loss') for m in messages[:2])
  originals[name]=messages[:2];targets[name]=messages[2];tools[name]=json.loads(row['tools_json'])
  text,_=pangolin_render(messages[:3],tools[name]);assert text.rstrip('\n')==row['text'][:span[1]]
  texts[name]=row['text'][span[0]:span[1]]
 family=views['original']['semantic_family_id'];counts['roots']+=1
 for name in ['original','counterfactual','invariance']:
  append(rid,family,name,originals[name],targets[name],tools[name],True,views[name]['record_id'])
 if texts['original']!=texts['counterfactual']:
  counts['discriminating_first_decision_pairs']+=1
  for name,other in [('original','counterfactual'),('counterfactual','original')]:
   append(rid,family,name,originals[name],targets[other],tools[name],False,views[name]['record_id'])
 else:counts['same_first_action_pair_not_scored']+=1
 public=json.loads(views['original']['public_episode_json'])
 for condition,field in [('no_skill','skills'),('no_evidence','context')]:
  altered=copy.deepcopy(public);altered[field]=[];prefix=actor_messages(altered)
  assert len(prefix)==2
  append(rid,family,condition,prefix,targets['original'],tools['original'],True,views['original']['record_id'])
 append(rid,family,'answer_only',[],targets['original'],[],True,views['original']['record_id'])
path=s/'context-probes.parquet';assert not path.exists()
pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd')
save_new(s/'context-probe-plan.json',{'counts':dict(counts),'rows':len(rows),'sources':inputs,'file_sha256':sha_file(path),'producer_sha256':sha_file(__file__),'scope':'all 2000 authored TRAIN roots; first accepted assistant decision only; conditional scoring, not free-generation accuracy','no_skill_meaning':'remove library text while retaining tool schema and evidence','no_evidence_meaning':'remove public documents, retain question/skills/tools; gold likelihood sensitivity only, not a correct response under missing evidence','answer_only_meaning':'bare assistant target; IFD-like diagnostic, not the full original IFD training method','wrong_candidates_are_evaluation_only':True,'training_data_created':0,'heldout_used':False})
print(json.dumps({'counts':dict(counts),'rows':len(rows)}),flush=True)
