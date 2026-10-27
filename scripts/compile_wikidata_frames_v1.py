"""Compile frozen singleton selections directly from lossless source frames."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from barbet_inference import candidate_archive as archive
from scripts import build_memory_sft_v2 as base
from scripts.prepare_v26_answer_candidates import canonicalize
from scripts import finalize_wikidata_singletons_v1 as selected
from scripts import normalize_wikidata_relations_v1 as source

STATE=selected.STATE
OUT=selected.OUT


def init():
    global SELECTION
    base.init_tokenizer()
    SELECTION=sqlite3.connect(f'file:{STATE / "selection.sqlite"}?mode=ro',uri=True)


def job(spec):
    ident=f"{spec['file_id']:05d}";receipt=STATE/'compiled'/(ident+'.json')
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        for f in saved['files']:
            if base.file_sha(OUT/f['path'])!=f['sha256']:raise ValueError('completed array artifact differs')
        return saved
    if base.file_sha(spec['path'])!=spec['sha256']:raise ValueError('candidate archive changed')
    rows={cid:(rid,family,split) for cid,rid,family,split in SELECTION.execute(
        'SELECT candidate_identity,record_id,family_id,split FROM selected WHERE file_id=?',(spec['file_id'],))}
    if len(rows)!=spec['selected_rows']:raise ValueError('selected file cardinality differs')
    buffers={s:{n:[] for n in ['records','tokenized','minicpm5_tokenized']} for s in ['train','validation']}
    counts=Counter();rejected=[]
    for _,_,raw in archive.iter_records(spec['path']):
        row=canonicalize(json.loads(raw));decision=rows.pop(row['candidate_identity'],None)
        if decision is None:continue
        if row['record_id']!=decision[0]:raise ValueError('selected record identity differs')
        row['family_id'],row['split']=decision[1:]
        row['source_refs']['answer_selection']={'method':'only_eligible_answer','competitors':1,'ppl_measured':False}
        counts['seen']+=1
        try:record,pang,mini=base.compile_row(row)
        except base.Rejected as exc:
            counts['excluded:'+str(exc)]+=1;rejected.append({'record_id':row['record_id'],'reason':str(exc)});continue
        for name,value in [('records',record),('tokenized',pang),('minicpm5_tokenized',mini)]:buffers[row['split']][name].append(value)
        counts['accepted']+=1
        for key in ['pangolin_tokens','pangolin_supervised_tokens','minicpm_tokens','minicpm_supervised_tokens']:
            counts[key]+=record[key]
    if rows:raise ValueError('selected source pointers missing from complete archive')
    files=[];shard_bytes=0
    for split,configs in buffers.items():
        for name,values in configs.items():
            if not values:continue
            dest=OUT/'data'/name/(split+'-'+ident+'.parquet');dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise ValueError('partial compiled shard exists; do not overwrite')
            table=pa.Table.from_pylist(values,schema=None if name=='records' else base.TOKEN_SCHEMA)
            sink=pa.BufferOutputStream()
            pq.write_table(table,sink,compression='zstd',compression_level=6,row_group_size=128)
            payload=sink.getvalue();shard_bytes+=len(payload)
            if shard_bytes>spec['maximum_export_bytes']:
                raise ValueError('fixed complete-export shard allocation exceeded before writing payload')
            with dest.open('xb') as handle:handle.write(payload)
            if not pq.read_table(dest).equals(table):raise ValueError('complete exported array readback differs')
            files.append({'path':str(dest.relative_to(OUT)),'sha256':base.file_sha(dest),'bytes':dest.stat().st_size,
                          'rows':len(values),'split':split,'config':name})
    result={'shard':ident,'counts':dict(counts),'files':files,'rejected':rejected,
        'all_exported_arrays_readback':True,'source_shard_sha256':spec['sha256'],
        'direct_lossless_frame_reading':True,'no_full_selected_jsonl_copy':True}
    base.save(receipt,result);return result


def main(workers):
    if (STATE/'compile-summary.json').exists():raise ValueError('complete compilation already frozen')
    binding=json.loads((STATE/'ppl-score-binding.json').read_text())
    if base.file_sha(STATE/'selection.sqlite')!=binding['selection_database_sha256']:
        raise ValueError('frozen selection metadata changed')
    resources=json.loads(source.RESOURCES.read_text())['resources'];reserve=36*1024**3
    if source.processed_bytes()+reserve>resources['new_processed_cap_bytes'] or shutil.disk_usage(base.WORK).free<resources['minimum_free_disk_bytes']+reserve:
        raise ValueError('prospective complete array stage exceeds storage cap or free floor')
    jobs=json.loads((STATE/'compile-jobs.json').read_text())
    jobs=[{**spec,'maximum_export_bytes':(reserve-1024**3)//len(jobs)} for spec in jobs]
    policy={'driver_sha256':base.file_sha(Path(__file__)),'original_compile_row_sha256':base.file_sha(Path(base.__file__)),
        'archive_reader_sha256':base.file_sha(Path(archive.__file__)),
        'selection_binding_sha256':base.file_sha(STATE/'ppl-score-binding.json'),
        'reserved_compile_bytes':reserve,'no_truncated_text_or_arrays':True,'same_native_compiler_and_exact_table_readback':True}
    p=STATE/'framed-compile-policy.json'
    if p.exists() and json.loads(p.read_text())!=policy:raise ValueError('frame compilation policy changed')
    if not p.exists():base.save(p,policy)
    counts=Counter();written=0
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for n,result in enumerate(pool.map(job,jobs),1):
            counts.update(result['counts']);written+=sum(f['bytes'] for f in result['files'])
            if written>reserve:raise ValueError('compiled arrays exceed initial stage reservation; retain artifacts for resource review')
            print(json.dumps({'compiled_shards':n,'total':len(jobs),'accepted':counts['accepted'],'written_bytes':written}),flush=True)
    if counts['seen']!=json.loads((STATE/'dedup-summary.json').read_text())['selected']:
        raise ValueError('compile coverage differs from complete singleton selection')
    base.save(STATE/'compile-summary.json',dict(counts));print(json.dumps(dict(counts)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=16);args=parser.parse_args()
    if not 1<=args.workers<=24:raise ValueError('CPU worker bound')
    main(args.workers)
