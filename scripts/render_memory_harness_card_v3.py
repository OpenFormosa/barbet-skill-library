"""Dataset-level facts and limitations, separate from any single source wave."""
def render(counts,unique,code,wave,new_rows,source_note):
    configs='\n'.join(f'''- config_name: {name}
  data_files:
  - split: train
    path: data/wave-*/{name}/train-*.parquet
  - split: validation
    path: data/wave-*/{name}/validation-*.parquet''' for name in ['records','tokenized','minicpm5_tokenized'])
    source_table='\n'.join(f"| {key[7:]} | {value:,} |" for key,value in sorted(counts.items()) if key.startswith('source:'))
    length_table='\n'.join(f"| ≤ {int(key.split(':')[1]):,} | {value:,} |" for key,value in sorted(counts.items(),key=lambda kv:int(kv[0].split(':')[1]) if kv[0].startswith('length_bucket:') else 0) if key.startswith('length_bucket:'))
    target='The 10,000,000-episode count target has been reached; complete publication verification is reported separately in PUBLICATION.json.' if counts['accepted']>=10000000 else 'The target is 10,000,000 accepted new episodes and has not yet been reached.'
    return f'''---
license: other
license_name: mixed-source-licenses
license_link: https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/blob/main/NOTICE.md
task_categories:
- text-generation
language:
- en
- zh
configs:
{configs}
---

# Barbet memory, skill and harness SFT

**{counts['accepted']:,} new, accepted episodes**, **{unique['query_hash']:,} distinct normalized first queries**, and **{unique['family_id']:,} task families**. Active legacy rows: **0**. {target} Earlier releases remain available in Hub history. The current addition contributes {new_rows:,} episodes.

This dataset teaches use of supplied evidence, explicit skill instructions, typed tools, real observations and verifiable answers. Memory is placed before the answer, at the front or back of the user context, or exposed through registered tool observations. Different answer versions, memory positions, URI variants and dialogue prefixes are not counted as independent tasks.

## Standard answer selection

For multiple eligible answers to an identical complete input, select the minimum conditional supervised-answer PPL under frozen [Barbet v26](https://huggingface.co/voidful/barbet-skill/tree/67131067ff1825179d5e38e91d499159669e9c66). The denominator includes supervised assistant tokens and assistant EOS only. Prompts, memory, tool observations, unsupervised history and padding are excluded. Native unpadded batch size is 1, without truncation. Every competitor must have a finite score. Different literal inputs or observation conditions are not treated as interchangeable alternatives. Equal scores use a stable record-ID tie break.

**{counts['selection:minimum_v26_conditional_answer_ppl']:,} episodes** were selected from scored alternatives. **{counts['selection:only_eligible_answer']:,} episodes** had one eligible answer and are explicitly unscored. No PPL is invented for those rows. Quality and source-contract holds apply before admission; a newly held selected question is removed entirely without substituting a runner-up. Low PPL measures likelihood, not truth.

## Size and source mix

| Measure | Count |
|---|---:|
| Train episodes | {counts['split:train']:,} |
| Validation episodes | {counts['split:validation']:,} |
| Pangolin physical tokens | {counts['pangolin_tokens']:,} |
| Pangolin supervised tokens | {counts['pangolin_supervised_tokens']:,} |
| MiniCPM5 physical tokens | {counts['minicpm_tokens']:,} |
| MiniCPM5 supervised tokens | {counts['minicpm_supervised_tokens']:,} |
| Assistant turns | {counts['assistant_turns']:,} |
| Tool observations | {counts['tool_observations']:,} |

| Source | Accepted episodes |
|---|---:|
{source_table}

The following buckets use the larger of the two complete native training lengths. Each row belongs to its smallest fitting bucket. Dataset name does not imply that every example is long-context; unrelated padding is not added.

| Upper token bound | Episodes |
|---|---:|
{length_table}

## Current source addition

{source_note}

## Training and inference interfaces

`records` contains portable messages, tool schemas, full native text, provenance, exact target spans and memory metadata. `tokenized` contains complete Pangolin IDs and assistant-only labels. `minicpm5_tokenized` contains complete native MiniCPM5-2B IDs and labels. Token IDs are tokenizer-specific. Memory, prompts and observations have loss masked out; only supervised assistant targets train the model.

The [skill library and harness](https://github.com/OpenFormosa/barbet-skill-library/tree/{code}) provide evidence retrieval, typed registered callbacks and bounded refinement from real verifier feedback. `barbet_inference.dataset_adapter.prepare_dataset_request` constructs answer-free Chat or exact native Completion input, excluding later target turns and provenance-only computation traces. For finite-relation records, explicitly call `barbet_inference.relational_dataset.relation_tool_callbacks(row)` to restore the bounded, read-only `relation_lookup` callback. Stored source text never automatically registers or executes code. The optional local finite-query verifier is separate from the model prompt.

All retained records and all token/label arrays are read back in full. Answer-free Qwen Chat and Completion requests are measured with its actual served tokenizer and 1,000,000-token context limit. Native training and MiniCPM5-2B checks include a 4,096-token output reserve within 131,072 tokens. These format and budget checks do not certify a Barbet inference backend. No truncation is used. Public request-proof record IDs exactly equal accepted record IDs.

## Scope and evaluation limits

Family and query folds are preserved across published waves; exact prior-question overlaps and recorded first holds are excluded. New graph tasks use the source training graph only. Their SFT train/validation split is by anchor family and can share other context entities or facts; it is **not** an entity-disjoint graph evaluation. Exact local benchmark text checks do not establish exhaustive semantic decontamination.

Mechanical validation, source-reported verification, exact finite-record query verification and sampled semantic diagnosis are distinct evidence tiers. First non-ACCEPT cases remain held, including incomplete reviews and contradictory reviewer complaints. Diagnostic samples are not corpus accuracy estimates. Source-code examples were not executed, and source assertions are not certified as current world truth. A large structured-query cohort can dominate raw row counts; training mixtures should be controlled by source, task type and supervised-token budget rather than assuming uniform rows are equally informative.

No model weights were trained in this rebuild. No new model capability gain, small-model/9B equivalence, or benchmark improvement is claimed. Exact procedural targets are data construction results, not model benchmark scores. Iterative refinement records require actual feedback; graph query rows without an error/refinement trajectory are not counted as refinement training.

Licenses remain per source and per record. See [NOTICE.md](NOTICE.md), `license`, `source_refs_json`, and the [current release report](reports/{wave}/report.html). CC-BY-SA terms remain attached to applicable adapted records; the collection does not relicense all sources under one permissive license. `PUBLICATION.json` binds the data revision, library commit and full anonymous SHA-256 readback. Readback status is recorded only after the actual download and checks complete.
'''
