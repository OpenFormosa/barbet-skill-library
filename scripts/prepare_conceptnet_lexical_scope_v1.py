"""Select a lexical-source scope and one anchor per base concept, without SFT."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3

from scripts import build_memory_sft_v2 as base
from scripts.inventory_conceptnet57_english_v1 import STATE as INVENTORY
from scripts.normalize_ifm_math_memory_v1 import processed_bytes

STATE=INVENTORY.parent/'lexical-scope-v1'
ALLOWED=['/d/wiktionary/en','/d/wiktionary/fr','/d/wiktionary/de','/d/wordnet/3.1']


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE/'policy.json').exists():raise ValueError('lexical source scope already started')
    source_db=INVENTORY/'english-assertions.sqlite';summary=json.loads((INVENTORY/'summary.json').read_text())
    if base.file_sha(source_db)!=summary['database_sha256']:raise ValueError('source assertion inventory differs')
    resources=json.loads((base.REPO/'configs/memory_sft_v2_scale_10m_resources_v3.json').read_text())['resources'];reserve=3*1024**3
    if processed_bytes()+reserve>resources['new_processed_cap_bytes'] or shutil.disk_usage(base.WORK).free<resources['minimum_free_disk_bytes']+reserve:
        raise ValueError('lexical scope storage bound')
    base.save(STATE/'policy.json',{'driver_sha256':base.file_sha(Path(__file__)),
        'source_inventory_sha256':base.file_sha(INVENTORY/'summary.json'),'source_database_sha256':summary['database_sha256'],
        'allowed_source_datasets':ALLOWED,'reserved_stage_bytes':reserve,
        'source_scope':'Wiktionary and WordNet lexical assertions only. Crowdsourced OMCS/Verbosity facts and other source families remain outside this prospective cohort. Source records are still evidence within a finite snapshot, not independently established world truth.',
        'anchor_identity':'Group subject URI variants by /c/en/<base-concept>. Prefer the unqualified URI when present, otherwise the lexicographically first source subject URI. At most one prospective task per base concept; sense/POS suffixes are not additional independent anchors.',
        'source_senses_and_negative_relations_preserved_inside_facts':True,
        'accepted_training_rows':0,'model_requests':0,'source_programs_executed':False})
    original=sqlite3.connect(f'file:{source_db}?mode=ro',uri=True)
    db_path=STATE/'lexical-index.sqlite';db=sqlite3.connect(db_path)
    db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF')
    db.execute('CREATE TABLE facts(source_line INTEGER PRIMARY KEY,subject TEXT,relation TEXT,object TEXT)')
    counts=Counter();degrees=Counter();bases={};batch=[]
    sql="SELECT source_line,subject,relation,object,metadata_json FROM facts WHERE json_extract(metadata_json,'$.dataset') IN ("+','.join('?' for _ in ALLOWED)+')'
    for source_line,subject,relation,object_,metadata_raw in original.execute(sql,ALLOWED):
        metadata=json.loads(metadata_raw);counts['facts']+=1;counts['license:'+metadata['license']]+=1;counts['source:'+metadata['dataset']]+=1
        degrees[subject]+=1;base_uri='/'.join(subject.split('/')[:4]);previous=bases.get(base_uri)
        if previous is None or subject==base_uri or (previous!=base_uri and subject<previous):bases[base_uri]=subject
        batch.append((source_line,subject,relation,object_))
        if len(batch)>=25000:db.executemany('INSERT INTO facts VALUES(?,?,?,?)',batch);db.commit();batch=[]
    db.executemany('INSERT INTO facts VALUES(?,?,?,?)',batch);db.commit()
    db.execute('CREATE INDEX by_subject_relation ON facts(subject,relation,object)')
    db.execute('CREATE TABLE anchors(base_uri TEXT PRIMARY KEY,subject TEXT,degree INTEGER)')
    db.executemany('INSERT INTO anchors VALUES(?,?,?)',[(b,s,degrees[s]) for b,s in sorted(bases.items())]);db.commit()
    by_degree=Counter('one' if degrees[s]==1 else 'two_to_eight' if degrees[s]<=8 else 'nine_to_32' if degrees[s]<=32 else 'above_32' for s in bases.values())
    if db.execute('SELECT COUNT(*) FROM facts').fetchone()[0]!=counts['facts']:raise ValueError('source subset count mismatch')
    original.close();db.close()
    if db_path.stat().st_size>reserve:raise ValueError('lexical indexed storage exceeded bound')
    result={'counts':dict(counts),'source_subject_uris':len(degrees),'distinct_base_concept_anchors':len(bases),
        'collapsed_subject_uri_variants':len(degrees)-len(bases),'anchor_degree_counts':dict(by_degree),
        'bounded_anchor_capacity':sum(n for k,n in by_degree.items() if k!='above_32'),
        'database_sha256':base.file_sha(db_path),'database_bytes':db_path.stat().st_size,
        'policy_sha256':base.file_sha(STATE/'policy.json'),'accepted_training_rows':0,'model_requests':0,
        'new_task_contract_first_diagnosis_and_export_gates_still_required':True}
    base.save(STATE/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
