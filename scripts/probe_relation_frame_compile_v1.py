"""Check actual framed compilation against the unchanged native compiler."""
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from barbet_inference import candidate_archive as archive
from barbet_inference.memory import canonical
from scripts import build_memory_sft_v2 as base
from scripts.prepare_v26_answer_candidates import canonicalize
from scripts import compile_wikidata_frames_v1 as compiler
from scripts import wikidata_relation_adapter_v2 as adapter
from scripts.diagnose_wikidata_relations_v1 import STATE as DIAG
from barbet_inference.library import SkillLibrary

STATE=adapter.INVENTORY.parent/'framed-compile-probe-v1'


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE/'selection.sqlite').exists():raise ValueError('framed compile probe already started')
    audit=json.loads((DIAG/'semantic-audit-v1/summary.json').read_text())
    good={r['record_id'] for r in audit['results'] if r['verdict']=='ACCEPT'};chosen={}
    for line in (DIAG/'normalized/pilot-relations.jsonl').open():
        row=json.loads(line)
        if row['record_id'] in good:chosen.setdefault(row['domain'],row)
    if set(chosen)!={'reading','tool_use'}:raise ValueError('both actual source layouts required')
    graph=adapter.SourceGraph();skills=SkillLibrary(base.REPO).render(adapter.SKILLS)
    db=sqlite3.connect(STATE/'selection.sqlite')
    db.execute('CREATE TABLE selected(candidate_identity TEXT PRIMARY KEY,record_id TEXT,family_id TEXT,split TEXT,file_id INTEGER)')
    expected={};path=STATE/'probe.bca';base.init_tokenizer()
    with archive.ArchiveWriter(path) as writer:
        for original in chosen.values():
            row=canonicalize(graph.make_row(int(original['source_task_key'][1:]),skills))
            writer.append((canonical(row)+'\n').encode())
            family=base.sha('family:'+row['source_family_id']);split='train'
            db.execute('INSERT INTO selected VALUES(?,?,?,?,?)',(row['candidate_identity'],row['record_id'],family,split,0))
            row['family_id']=family;row['split']=split
            row['source_refs']['answer_selection']={'method':'only_eligible_answer','competitors':1,'ppl_measured':False}
            expected[row['record_id']]=base.compile_row(row)
    db.commit();db.close()
    compiler.STATE=STATE;compiler.OUT=STATE/'export';compiler.init()
    report=compiler.job({'file_id':0,'path':str(path),'sha256':base.file_sha(path),'selected_rows':2,'maximum_export_bytes':4*1024**2})
    configs={'records':0,'tokenized':1,'minicpm5_tokenized':2};rows=0
    for item in report['files']:
        for actual in pq.read_table(compiler.OUT/item['path']).to_pylist():
            if actual!=expected[actual['record_id']][configs[item['config']]]:
                raise ValueError('actual framed export differs from unchanged native compiler output')
            rows+=1
    if rows!=6:raise ValueError('three-config exact probe coverage differs')
    result={'driver_sha256':base.file_sha(Path(__file__)),'compiler_sha256':base.file_sha(Path(compiler.__file__)),
        'samples':2,'layouts':['reading','tool_use'],'all_three_configs_equal_direct_native_compiler':True,
        'full_records_arrays_labels_and_inference_hashes_equal':True,'accepted_training_rows':0,
        'model_requests':0,'source_programs_executed':False}
    base.save(STATE/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
