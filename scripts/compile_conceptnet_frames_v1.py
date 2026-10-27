"""Use the verified framed compiler with a lexical-specific bounded allocation."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from barbet_inference import candidate_archive as archive
from barbet_inference.memory import canonical
from scripts import build_memory_sft_v2 as base
from scripts import compile_wikidata_frames_v1 as compiler
from scripts import finalize_conceptnet_singletons_v1 as selected
from scripts import normalize_conceptnet_relations_v1 as source
from scripts.index_conceptnet_relations_v1 import reserve
from scripts.prepare_v26_answer_candidates import canonicalize

STATE=selected.STATE
OUT=selected.OUT


def probe():
    state=source.STATE.parent/'framed-compile-probe-v1';state.mkdir(exist_ok=True)
    if (state/'selection.sqlite').exists():raise ValueError('first lexical framed probe already started')
    audit=json.loads((source.diagnosis.STATE/'semantic-audit-v1/summary.json').read_text())
    good={r['record_id'] for r in audit['results'] if r['verdict']=='ACCEPT'};chosen={}
    for row in map(json.loads,(source.diagnosis.STATE/'normalized/pilot-lexical.jsonl').open()):
        if row['record_id'] in good:chosen.setdefault(row['domain'],row)
    if set(chosen)!={'reading','tool_use'}:raise ValueError('both source layouts required')
    db=sqlite3.connect(state/'selection.sqlite')
    db.execute('CREATE TABLE selected(candidate_identity TEXT PRIMARY KEY,record_id TEXT,family_id TEXT,split TEXT,file_id INTEGER)')
    expected={};path=state/'probe.bca';base.init_tokenizer()
    with archive.ArchiveWriter(path) as writer:
        for original in chosen.values():
            row=canonicalize(original);writer.append((canonical(row)+'\n').encode())
            family=base.sha('family:'+row['source_family_id']);split='train'
            db.execute('INSERT INTO selected VALUES(?,?,?,?,?)',(row['candidate_identity'],row['record_id'],family,split,0))
            row['family_id']=family;row['split']=split
            row['source_refs']['answer_selection']={'method':'only_eligible_answer','competitors':1,'ppl_measured':False}
            expected[row['record_id']]=base.compile_row(row)
    db.commit();db.close();compiler.STATE=state;compiler.OUT=state/'export';compiler.init()
    report=compiler.job({'file_id':0,'path':str(path),'sha256':base.file_sha(path),'selected_rows':2,'maximum_export_bytes':4*1024**2})
    configs={'records':0,'tokenized':1,'minicpm5_tokenized':2};rows=0
    for item in report['files']:
        for actual in pq.read_table(compiler.OUT/item['path']).to_pylist():
            if actual!=expected[actual['record_id']][configs[item['config']]]:raise ValueError('lexical framed/direct arrays differ')
            rows+=1
    if rows!=6:raise ValueError('three-config coverage differs')
    result={'driver_sha256':base.file_sha(Path(__file__)),'original_framed_compiler_sha256':base.file_sha(Path(compiler.__file__)),
        'samples':2,'all_three_configs_and_full_records_equal_direct_compiler':True,
        'accepted_training_rows':0,'model_requests':0}
    base.save(state/'summary.json',result);print(json.dumps(result),flush=True)


def main(workers):
    if (STATE/'compile-summary.json').exists():raise ValueError('full lexical compilation already frozen')
    binding=json.loads((STATE/'ppl-score-binding.json').read_text())
    if base.file_sha(STATE/'selection.sqlite')!=binding['selection_database_sha256']:raise ValueError('selected metadata changed')
    probe_path=source.STATE.parent/'framed-compile-probe-v1/summary.json';proof=json.loads(probe_path.read_text())
    if proof['driver_sha256']!=base.file_sha(Path(__file__)) or proof['original_framed_compiler_sha256']!=base.file_sha(Path(compiler.__file__)):
        raise ValueError('first lexical framed probe implementation changed')
    amount=4*1024**3;reserve('full-compile',amount)
    jobs=json.loads((STATE/'compile-jobs.json').read_text())
    jobs=[{**spec,'maximum_export_bytes':(amount-256*1024**2)//len(jobs)} for spec in jobs]
    policy={'driver_sha256':base.file_sha(Path(__file__)),'original_framed_compiler_sha256':base.file_sha(Path(compiler.__file__)),
        'original_compile_row_sha256':base.file_sha(Path(base.__file__)),'archive_reader_sha256':base.file_sha(Path(archive.__file__)),
        'selection_binding_sha256':base.file_sha(STATE/'ppl-score-binding.json'),
        'real_source_frame_probe_sha256':base.file_sha(probe_path),'reserved_compile_bytes':amount,
        'no_truncated_text_or_arrays':True,'same_native_compiler_and_exact_table_readback':True}
    path=STATE/'framed-compile-policy.json'
    if path.exists() and json.loads(path.read_text())!=policy:raise ValueError('frozen lexical compilation policy changed')
    if not path.exists():base.save(path,policy)
    compiler.STATE=STATE;compiler.OUT=OUT;counts=Counter();written=0
    with ProcessPoolExecutor(max_workers=workers,initializer=compiler.init) as pool:
        for n,result in enumerate(pool.map(compiler.job,jobs),1):
            counts.update(result['counts']);written+=sum(f['bytes'] for f in result['files'])
            if written>amount:raise ValueError('lexical export reservation exceeded')
            print(json.dumps({'compiled_shards':n,'total':len(jobs),'accepted':counts['accepted'],'written_bytes':written}),flush=True)
    if counts['seen']!=json.loads((STATE/'dedup-summary.json').read_text())['selected']:
        raise ValueError('complete lexical compile coverage differs')
    base.save(STATE/'compile-summary.json',dict(counts));print(json.dumps(dict(counts)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['probe','compile']);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if not 1<=args.workers<=16:raise ValueError('CPU worker bound')
    if args.phase=='probe':probe()
    else:main(args.workers)
