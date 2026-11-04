"""Anonymous Hub config integration controls; exhaustive arrays were checked separately."""
import gc,json,sys
from pathlib import Path
import yaml
from urllib.request import urlopen
from datasets import load_dataset
from barbet_pipeline.common import save_new
from barbet_pipeline.full_requests import prepare_full_request

revision,output=sys.argv[1:]
repo='OpenFormosa/barbet-long-context-sft'
with urlopen(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md',timeout=120) as response:
    card=response.read().decode()
front=yaml.safe_load(card.split('---',2)[1]);controls=[];identities={}
for config in front['configs']:
    name=config['config_name']
    for entry in config['data_files']:
        split=entry['split']
        ds=load_dataset(repo,name,revision=revision,split=split,streaming=True,token=False)
        iterator=iter(ds);row=next(iterator)
        if name.startswith('full_v4_') and name.endswith('_records'):
            assert row['schema_version']=='barbet-full-reprocess-v4'
            for mode in ['chat','completion']:prepare_full_request(row,model='configured-server-model',mode=mode)
            status=name[len('full_v4_'):-len('_records')]
            assert row['processing_status']==status
            identities[(status,split,'records')]=row['record_id']
        elif name.startswith('full_v4_') and name.endswith('_tokenized'):
            assert len(row['input_ids'])==len(row['labels'])==row['pangolin_tokens']
            assert sum(x!=-100 for x in row['labels'])==row['pangolin_supervised_tokens']
            status=name[len('full_v4_'):-len('_tokenized')]
            identities[(status,split,'tokenized')]=row['record_id']
        elif name=='full_v4_dispositions':
            assert split=='corpus' and row['parent_record_id'] and row['status']
        elif name=='full_v4_gpu_measurements':
            from barbet_pipeline.gpu_release import validate_measurement
            validate_measurement(row)
            assert row['split']==split and row['parent_record_id'] and row['source']
            assert row['model_revision']=='67131067ff1825179d5e38e91d499159669e9c66'
            assert row['automatic_training_admission'] is False
        else:assert name in {'semantic_v3_records','semantic_v3_tokenized'} and row['record_id']
        controls.append({'configuration':name,'split':split,'first_row_loaded':True,'anonymous':True})
        del row,iterator,ds;gc.collect()
        print(json.dumps(controls[-1]),flush=True)
for (status,split,kind),rid in identities.items():
    assert identities[(status,split,'records')]==identities[(status,split,'tokenized')]
save_new(Path(output),{'revision':revision,'all_configs_and_splits_loaded':True,'controls':controls,
    'first_record_tokenized_identities_match':True,'scope':'config/schema/inference adapter integration controls only; every source row and saved token array was verified by the processing workers'})
