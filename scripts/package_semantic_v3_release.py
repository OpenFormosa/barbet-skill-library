"""Build public metadata only after all learner shards have been verified.

Private oracle and held-out records are not accepted inputs. The final manifest
indexes newly published files; the inherited source bank stays at its parent.
"""
from pathlib import Path
import json
import subprocess
import sys
import yaml
from barbet_pipeline.common import read_json,save_new,sha_file


def package(state,repo,package_name='release-package'):
    state,repo=Path(state),Path(repo)
    if Path(package_name).name!=package_name:raise ValueError('Package name must be a single directory component')
    out=state/package_name
    if out.exists():raise FileExistsError(out)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise ValueError('Commit the implementation before packaging')
    data=[]
    for label in ['source-index','learner-records','faded-binding']:
        plan=read_json(state/'publication'/(label+'-plan.json'))
        receipts=list((state/'publication'/(label+'-batches')).glob('*.json'))
        verified={p for r in receipts for p in read_json(r)['paths']}
        assert verified=={x['remote'] for x in plan['items']}
        data+=plan['items']
    assert len(data)==1754 and len({x['remote'] for x in data})==1754
    remap={x['local']:x['remote'] for x in data}
    def public(value):
        if isinstance(value,dict):return {k:public(v) for k,v in value.items()}
        if isinstance(value,list):return [public(v) for v in value]
        if isinstance(value,str) and value.startswith('/root/'):
            return remap.get(value,'local-path-omitted/'+Path(value).name)
        return value
    metadata=[]
    def add(local,remote):
        metadata.append({'local':str(local),'remote':remote,'sha256':sha_file(local),'bytes':local.stat().st_size})
    def write(name,value):
        path=out/'metadata'/name;save_new(path,public(value));add(path,'semantic-v3/metadata/'+name)
    for cohort in ['corpus','reading-natural-v3','pilot','faded-binding','pilot-release']:
        write(cohort+'-summary.json',read_json(state/cohort/'summary.json'))
    for cohort in ['reading-natural-v3','pilot','faded-binding']:
        for check in ['exact-validation.json','transport-validation.json','transport-validation.json.live-controls.json']:
            write(cohort+'-'+check,read_json(state/cohort/check))
    for name in ['release-summary','final-semantic-audit','cost-receipt','teacher-shutdown-complete']:
        write(name+'.json',read_json(state/(name+'.json')))
    write('A-ordinary.json',read_json(state/'pilot/A_ordinary-manifest.json'))
    write('B-contrasts-and-fading.json',read_json(state/'pilot-release/B_semantic_contrasts_and_fading-manifest.json'))
    for name in ['semantic_curriculum_v3.json','semantic_v3_sources.lock.json','semantic_v3_environment.lock.json',
                 'semantic_v3_model_manifest.json','taiwan-civic-policy-v1.json','value-alignment-policy-v1.json']:
        write(name,read_json(repo/'configs'/name))
    source=read_json(state/'reading-natural-v3/plan.json')['source']
    write('natural-source-provenance.json',{'source':{k:v for k,v in source.items() if k!='local'},
        'license':'cc-by-sa-4.0','attribution':'SQuAD 2.0, Rajpurkar et al.; passages retain their source context',
        'changes':'natural source-only answer rendering, optional versioned skill and front/back memory placement',
        'source_answers_preserved':True,'upstream_gold_validation_not_world_truth':True})
    add(repo/'docs/reports/semantic-curriculum-v3/final.html','semantic-v3/report.html')
    add(repo/'docs/semantic-curriculum-v3.md','semantic-v3/method.md')
    configs=[]
    for name,directory in [('semantic_v3_records','pilot/records'),('semantic_v3_tokenized','pilot/tokenized'),
                           ('natural_reading_v3_records','natural-reading/records'),('natural_reading_v3_tokenized','natural-reading/tokenized')]:
        entry={'config_name':name,'data_files':[{'split':'train','path':'semantic-v3/'+directory+'/train-*.parquet'},
            {'split':'validation','path':'semantic-v3/'+directory+('/dev-*.parquet' if directory.startswith('pilot') else '/validation-*.parquet')}]}
        if name=='semantic_v3_records':entry['default']=True
        configs.append(entry)
    configs.append({'config_name':'source_curriculum_index_v3','data_files':[{'split':'corpus','path':'semantic-v3/source-index/*.parquet'}]})
    front={'language':['zh','en'],'license':'other','license_name':'per-source-licenses-preserved',
        'license_link':'https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/blob/semantic-curriculum-v3-20261104/README.md#licensing','task_categories':['text-generation'],
        'tags':['barbet','semantic-curriculum','tool-use','skill','experimental'],'configs':configs}
    text='''# Barbet long-context SFT — semantic curriculum v3 candidate

This branch contains **121,040 validated experimental learner views** and a new curriculum index over **10,018,967 inherited source records**. No new model training was run. Data correctness does not establish learnability, transfer, retention, or native1M generation.

The original release is preserved at [b88f4ff7171bb6fb8c1b775ae21de814ea2123fb](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/b88f4ff7171bb6fb8c1b775ae21de814ea2123fb). This candidate does not replace `main`. Publication is complete only when `semantic-v3/release-manifest.json` exists and `SEMANTIC_V3_STAGING_INCOMPLETE.txt` is absent. Even after publication, **training_ready remains false** until model, trainer, curriculum and evaluation gates are met.

| Configuration | Train | Validation | Use |
|---|---:|---:|---|
| `semantic_v3_records` (default) | 6,286 views / 2,000 roots | 629 views / 200 roots | Seven atomic semantic families, discriminating counterfactuals, invariance and faded binding |
| `semantic_v3_tokenized` | 6,286 | 629 | Exactly matching Pangolin IDs and unshifted assistant-only labels |
| `natural_reading_v3_records` | 111,259 | 2,866 | Reprocessed existing SQuAD questions, natural source-only answers |
| `natural_reading_v3_tokenized` | 111,259 | 2,866 | Exactly matching Pangolin IDs and labels |
| `source_curriculum_index_v3` | — | — | 10,018,967 rows in `corpus`; routing metadata only, never automatic TRAIN admission |

The semantic records retain internal `split="dev"` for the Hub `validation` split. Calibration and sealed TEST each contain 200 roots and remain private. Tokenized configurations are alternate representations of the same records, **not additional examples**. The source index includes both inherited TRAIN and validation; its split is deliberately named `corpus`.

```python
from datasets import load_dataset
data = load_dataset(
    "OpenFormosa/barbet-long-context-sft",
    "semantic_v3_records",
    revision="semantic-curriculum-v3-20261104",  # pin its immutable commit for experiments
    split="train",
)
```

Do not concatenate every config or train directly from the legacy source bank. The proposed 50/50 general/semantic supervised-token mixture is not validated. Semantic TRAIN has only **462,008 supervised tokens per pass**; the proposed 2M/8M checkpoints were not run or filled by repeated exposure. Ordinary A has 2,000 views / 148,328 supervised tokens; contrast-and-fading B has 6,286 / 462,008. These are data menus, not matched-budget experiment results.

## Changes and verification

- Natural reading removes compulsory JSON/tool wrappers, preserves pinned source question/passage/gold and article split, and places evidence before or after the question. 75,609 are answerable and 38,516 source-unanswerable. One over-budget view was held; no silent truncation. These are existing questions, not new independent sources.
- Semantic tasks cover comparison, ranking, typed binding, units, current fictional rule versions, evidence sufficiency and execution states. Every root has an answer-changing counterfactual and answer-preserving invariant. Binding additionally asks for a display name and field meaning, requiring the model to find the actual tool ID/field code.
- **121,040 public rows** passed exact text, Pangolin IDs, unshifted labels, EOS and causal-prefix reconstruction. All public Chat/Completion decision prefixes passed served-tokenizer budgeting with live token-ID controls and a 4,096-token output reserve. Qwen tokenization is only a transport audit, not the Barbet training tokenizer.
- All **8,172 semantic views**, including private splits, passed grouped split/variant checks; **7,355 tool calls** were reexecuted from public state. 7,059 views use independent numeric/typed/state checks; 1,113 use source/state consistency checks. **316 program tests passed**.
- Raw proposals, host validation and effects stay distinct. Only independently accepted assistant targets and EOS receive loss; documents, skills and tool observations do not. Tools are actual deterministic executors or an explicitly versioned local state simulator, not fabricated external receipts.
- Frozen-v26 target selection is retained for eligible alternatives under the same full input. New deterministic singleton targets are explicitly unscored (`v26_ppl=null`). PPL is not used as truth or teaching-value evidence.
- Existing **310,662 holds remain excluded**. No rejected job was reopened. Oracle data and private calibration/TEST records are not in this repository branch.

The complete public learner export contains **106,824,940 input tokens** and **2,042,774 supervised tokens**. Every newly published file has SHA-256 and size in the [release manifest](semantic-v3/release-manifest.json); upload batches were anonymously downloaded and byte-verified.

## Limits and next gates

This is a short semantic pilot, with 8,192 total-token budget and 4,096 output reserve, not a new 128K–1M SFT curriculum. Atomic program graphs are shared across splits, and question frames are reused across procedural instances. It does not establish unseen-composition, schema or natural-source transfer. Reading source-gold checks do not establish world truth or learning value. The 516-item remote DEV raw outputs were unavailable; complete nearest-neighbor decontamination is not claimed.

The current base-named HF revision is declared SFT in its own release manifest; its earlier parent also has attention/surface-binding adaptation. A suitable training parent remains unselected. Native Barbet smoke used only 14/290-token prompts and 12 generated tokens, both hitting the output limit. It is not Megatron/CP/recompute parity, task success, or 1M serving certification.

The general ability retention mixture, independent transfer tests, matched A/B training and bare/same-harness four-cell evaluation remain pending. No candidate has been promoted as a better checkpoint. Qwen/MiniCPM parity is **NOT_ESTABLISHED**.

## Values and licensing

The existing Taiwan sovereignty, people's self-determination and universal human-rights policies remain in force, with factual and quotation fidelity. The 933 existing civic records remain in the preserved source bank. No new political targets or newly trained alignment improvement are claimed. See [Taiwan civic policy](semantic-v3/metadata/taiwan-civic-policy-v1.json) and [value policy](semantic-v3/metadata/value-alignment-policy-v1.json).

## Licensing

New authored semantic cases use Apache-2.0. Reprocessed SQuAD 2.0 material retains **CC BY-SA 4.0**, its original attribution and pinned provenance; see [source receipt](semantic-v3/metadata/natural-source-provenance.json). The inherited sources retain their individual licenses. The whole multi-source dataset is not relicensed as Apache-2.0. Records carry per-source license/provenance fields.

## Reports and code

[Full Traditional Chinese report](semantic-v3/report.html) · [Method and limits](semantic-v3/method.md) · [Aggregate results](semantic-v3/metadata/release-summary.json) · [A menu](semantic-v3/metadata/A-ordinary.json) · [B menu](semantic-v3/metadata/B-contrasts-and-fading.json)

IMPLEMENTATION_LINK
'''
    text=text.replace('IMPLEMENTATION_LINK',f'Implementation pinned at [OpenFormosa/barbet-skill-library@{commit}](https://github.com/OpenFormosa/barbet-skill-library/tree/{commit}).')
    card=out/'README.md';card.write_text('---\n'+yaml.safe_dump(front,sort_keys=False,allow_unicode=True)+'---\n\n'+text)
    add(card,'README.md')
    all_items=data+metadata
    manifest={'schema_version':'barbet-semantic-release-v3','status':'PUBLISHED_CANDIDATE_DATA_NOT_MODEL_TRAINING',
        'requires_staging_marker_absent':True,'training_ready':False,'parent_revision':'b88f4ff7171bb6fb8c1b775ae21de814ea2123fb',
        'candidate_branch':'semantic-curriculum-v3-20261104','implementation_commit':commit,
        'public_learner_rows':121040,'inherited_source_index_rows':10018967,
        'files':[{'path':x['remote'],'sha256':x['sha256'],'bytes':x['bytes']} for x in all_items],
        'hash_scope':'new data plus documentation; excludes this manifest itself and the later completion receipt',
        'private_oracle_or_heldout_records_published':False,'learning_transfer_retention':'NOT_RUN'}
    path=out/'release-manifest.json';save_new(path,manifest);add(path,'semantic-v3/release-manifest.json')
    save_new(out/'upload-plan.json',{'items':metadata})
    print(json.dumps({'implementation_commit':commit,'data_files':len(data),'metadata_files':len(metadata),'manifested_files':len(all_items)}))


if __name__=='__main__':package(*sys.argv[1:])
