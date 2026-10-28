"""Check stored civic targets against the real live Barbet prompt path, no model."""
import json
from types import SimpleNamespace

import pyarrow.parquet as pq
from tokenizers import Tokenizer
from barbet_inference import ContextAgent, SkillLibrary
from scripts.build_taiwan_civic_v1 import STATE, DEST, base, prior


def main():
    output=STATE/'live-prefix-equivalence.json'
    if output.exists():raise ValueError('First civic live prefix check already recorded')
    tokenizer=Tokenizer.from_file(str(base.WORK/'data/blobs/tokenizer/tokenizer.json'))
    library=SkillLibrary(base.REPO)
    library.render(['taiwan-civic-values','grounded-evidence-answer'])
    agent=ContextAgent(SimpleNamespace(tokenizer=tokenizer,max_model_len=131072),library=library,
        skills=['taiwan-civic-values','grounded-evidence-answer'],prompt_format='barbet_portable_v1')
    checked=0
    for item in prior.read(STATE/'new-cohort-summary.json')['files']:
        if '/records/' not in item['path']:continue
        tokens={t['record_id']:t for t in pq.read_table(DEST/item['path'].replace('/records/','/tokenized/')).to_pylist()}
        for row in pq.read_table(DEST/item['path'],columns=['record_id','messages_json']).to_pylist():
            token=tokens[row['record_id']];first=next(i for i,l in enumerate(token['labels']) if l!=-100)
            actual=agent._tokens(json.loads(row['messages_json'])[:-1])
            if actual!=token['input_ids'][:first]:raise ValueError('Live vs stored causal prefix mismatch')
            checked+=1
    if checked!=933:raise ValueError('Incomplete coverage')
    result={'records':checked,'all_live_context_agent_prefix_ids_equal_stored_supervised_prefix':True,
        'skill_count':len(library.list()),'no_model_generations_or_weight_updates':True,
        'not_model_capability_or_political_alignment_measurement':True}
    prior.save(output,result);prior.save(DEST/'reports/taiwan-civic-v1/live-prefix-equivalence.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
