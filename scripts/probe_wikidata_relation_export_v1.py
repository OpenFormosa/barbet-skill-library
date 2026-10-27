"""Measure real storage and restore tool state without admitting pilot rows."""
import json
from pathlib import Path
from collections import Counter
import random

import pyarrow as pa
import pyarrow.parquet as pq

from barbet_inference import candidate_archive as archive
from barbet_inference.library import SkillLibrary
from barbet_inference.memory import canonical
from barbet_inference.relational_dataset import restore_relational_memory
from barbet_inference.sft import Rejected
from scripts import build_memory_sft_v2 as base
from scripts import wikidata_relation_adapter_v2 as adapter
from scripts import diagnose_wikidata_relations_v1 as diagnosis

STATE=adapter.INVENTORY.parent/'export-probe-v1'


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE/'plan.json').exists():raise ValueError('first export probe already started')
    graph=adapter.SourceGraph();skills=SkillLibrary(base.REPO).render(adapter.SKILLS);base.init_tokenizer()
    base.save(STATE/'plan.json',{'driver_sha256':base.file_sha(Path(__file__)),
        'adapter_v2_sha256':base.file_sha(Path(adapter.__file__)),
        'first_adapter_v1_sha256':base.file_sha(Path(adapter.original.__file__)),
        'relational_restore_sha256':base.file_sha(base.REPO/'barbet_inference/relational_dataset.py'),
        'scope':'Metadata-only tool-state restoration check on all first pilot rows, then a seeded 1024-anchor storage measurement. No semantic replay, no publication, no admitted rows.',
        'source_programs_executed':False,'model_requests':0,'accepted_training_rows':0})
    equal=0;callback_count=0
    for line in (diagnosis.STATE/'normalized/pilot-relations.jsonl').open():
        old=json.loads(line);new=graph.make_row(int(old['source_task_key'][1:]),skills)
        for key in ['messages','tools','record_id','query_hash','family_id','skills']:
            if old[key]!=new[key]:raise ValueError('portable metadata changed first pilot model inputs or targets')
        record,pi,mi=base.compile_row(new);old_record,old_pi,old_mi=base.compile_row(old)
        if pi!=old_pi or mi!=old_mi or record['text']!=old_record['text'] or record['mini_text']!=old_record['mini_text']:
            raise ValueError('metadata annotation changed complete native text/tokens/labels')
        memory=restore_relational_memory(new)
        for call in new['source_refs']['actual_tool_calls']:
            response=memory.lookup(**call['arguments'])
            if base.sha(canonical(response))!=call['observation_sha256']:
                raise ValueError('portable restored callback differs from actual observation')
            callback_count+=1
        equal+=1
    rng=random.Random(2026102704);anchors=rng.sample(range(len(graph.heads)),1024)
    buffers={'records':[],'tokenized':[],'minicpm5_tokenized':[]};counts=Counter();raw_bytes=0
    with archive.ArchiveWriter(STATE/'candidates.bca') as writer:
        for index in anchors:
            try:
                row=graph.make_row(int(graph.heads[index]),skills)
                memory=restore_relational_memory(row)
                sql=diagnosis.independent_sql_target(row['source_refs']['query'],memory.facts)
                if canonical(sql)!=row['messages'][-1]['content']:raise ValueError('independent SQL target differs')
                row['split']='train'
                record,pang,mini=base.compile_row(row)
            except Rejected as exc:counts['held:'+str(exc)]+=1;continue
            raw=(canonical(row)+'\n').encode();writer.append(raw);raw_bytes+=len(raw)
            for name,value in [('records',record),('tokenized',pang),('minicpm5_tokenized',mini)]:buffers[name].append(value)
            counts['rows']+=1;counts['pangolin_tokens']+=record['pangolin_tokens'];counts['minicpm_tokens']+=record['minicpm_tokens']
            counts['schema:'+row['source_refs']['query_schema_id']]+=1
    files=[]
    for name,rows in buffers.items():
        path=STATE/(name+'.parquet');schema=None if name=='records' else base.TOKEN_SCHEMA
        pq.write_table(pa.Table.from_pylist(rows,schema=schema),path,compression='zstd')
        files.append({'config':name,'bytes':path.stat().st_size,'sha256':base.file_sha(path)})
    rows=counts['rows'];result={'counts':dict(counts),'files':files,'source_json_payload_bytes':raw_bytes,
        'candidate_archive_bytes':(STATE/'candidates.bca').stat().st_size,'prior_pilot_model_input_equivalence_rows':equal,
        'prior_actual_observations_restored_identically':callback_count,'all_new_targets_independently_sql_checked':True,
        'measured_total_bytes_per_row':((STATE/'candidates.bca').stat().st_size+sum(f['bytes'] for f in files))/rows,
        'naive_bytes_for_4554029_anchors':int(((STATE/'candidates.bca').stat().st_size+sum(f['bytes'] for f in files))/rows*4554029),
        'projection_caveat':'A seeded 1024-anchor measurement, not an upper bound or full-corpus count. Normalized identity metadata and budget proofs need additional storage. Copying payload files would double-count physical storage; future safe hardlinks may retain exact bytes after all gates.',
        'accepted_training_rows':0,'model_requests':0,'plan_sha256':base.file_sha(STATE/'plan.json')}
    base.save(STATE/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
