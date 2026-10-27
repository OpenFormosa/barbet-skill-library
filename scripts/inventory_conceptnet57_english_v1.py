"""Inventory original English assertions with exact per-edge provenance."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3

from scripts import build_memory_sft_v2 as base
from scripts.fetch_conceptnet57_primary_v1 import STATE as SOURCE
from scripts.normalize_ifm_math_memory_v1 import processed_bytes

STATE=SOURCE/'english-inventory-v1'


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE/'policy.json').exists():raise ValueError('source inventory already started; retain first state')
    lock=json.loads((SOURCE/'source-lock.json').read_text())
    resources=json.loads((base.REPO/'configs/memory_sft_v2_scale_10m_resources_v2.json').read_text())['resources'];reserve=8*1024**3
    used=processed_bytes();free=shutil.disk_usage(base.WORK).free
    if used+reserve>resources['new_processed_cap_bytes'] or free<resources['minimum_free_disk_bytes']+reserve:
        raise ValueError('source inventory exceeds prospective resource bound')
    if base.file_sha(lock['local'])!=lock['sha256']:raise ValueError('original source archive changed')
    base.save(STATE/'policy.json',{'driver_sha256':base.file_sha(Path(__file__)),
        'source_lock_sha256':base.file_sha(SOURCE/'source-lock.json'),'stage_reservation_bytes':reserve,
        'processed_bytes_before':used,'free_disk_bytes_before':free,
        'scope':'Original source rows where both concept URIs are English, all fact fields fit the bounded relational-memory contract, and explicit row license is CC-BY-SA 4.0 or CC-BY 4.0 with positive finite weight and recorded sources. Preserve exact URI/sense/polarity and source JSON; no claim that a source assertion is true in the world.',
        'held_other_row_licenses_not_relicensed':True,'accepted_training_rows':0,'source_programs_executed':False,'model_requests':0})
    db_path=STATE/'english-assertions.sqlite';db=sqlite3.connect(db_path)
    db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF');db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE facts(source_line INTEGER PRIMARY KEY, assertion_uri TEXT NOT NULL, relation TEXT NOT NULL, subject TEXT NOT NULL, object TEXT NOT NULL, metadata_json TEXT NOT NULL)')
    counts=Counter();licenses=Counter();datasets=Counter();relations=Counter();digest=hashlib.sha256();rows=[];decoded_bytes=0
    with gzip.open(lock['local'],'rb') as handle:
        for number,line in enumerate(handle,1):
            digest.update(line);decoded_bytes+=len(line);counts['raw_rows']+=1
            cells=line.rstrip(b'\r\n').split(b'\t')
            if len(cells)!=5:raise ValueError('original assertion row does not have five columns')
            if not cells[2].startswith(b'/c/en/') or not cells[3].startswith(b'/c/en/'):
                counts['outside_both_english_concepts']+=1;continue
            values=[s.decode('utf-8') for s in cells];assertion,relation,subject,object_,raw_metadata=values
            counts['both_english_rows']+=1
            metadata=json.loads(raw_metadata);license_name=metadata.get('license');licenses[str(license_name)]+=1
            if license_name not in {'cc:by-sa/4.0','cc:by/4.0'}:
                counts['held:unadmitted_explicit_row_license']+=1;continue
            if not isinstance(metadata.get('sources'),list) or not metadata['sources']:
                counts['held:missing_source_provenance']+=1;continue
            weight=metadata.get('weight')
            if type(weight) not in {int,float} or not math.isfinite(weight) or weight<=0:
                counts['held:nonpositive_or_invalid_weight']+=1;continue
            if any(not 0<len(v)<=256 for v in [relation,subject,object_]) or not relation.startswith('/r/'):
                counts['held:bounded_fact_contract']+=1;continue
            rows.append((number,assertion,relation,subject,object_,raw_metadata));counts['inventory_eligible_assertions']+=1
            relations[relation]+=1;datasets[str(metadata.get('dataset'))]+=1
            if len(rows)>=25000:
                db.executemany('INSERT INTO facts VALUES(?,?,?,?,?,?)',rows);db.commit();rows=[]
                if db_path.stat().st_size>reserve-1024**3:raise ValueError('bounded source inventory storage exhausted')
            if counts['both_english_rows']%1000000==0:
                print(json.dumps({'raw_rows_seen':number,'both_english_rows':counts['both_english_rows'],'eligible_assertions':counts['inventory_eligible_assertions']}),flush=True)
    if digest.hexdigest()!=lock['uncompressed_sha256'] or decoded_bytes!=lock['uncompressed_bytes'] or counts['raw_rows']!=lock['newline_count']:
        raise ValueError('full original assertion-stream checksum/count changed')
    db.executemany('INSERT INTO facts VALUES(?,?,?,?,?,?)',rows);db.commit()
    db.execute('CREATE INDEX by_subject_relation ON facts(subject,relation,object)')
    db.execute('CREATE INDEX by_object_relation ON facts(object,relation,subject)');db.commit()
    duplicate_groups=db.execute('SELECT COUNT(*) FROM (SELECT 1 FROM facts GROUP BY subject,relation,object HAVING COUNT(*)>1)').fetchone()[0]
    duplicate_uris=db.execute('SELECT COUNT(*) FROM (SELECT 1 FROM facts GROUP BY assertion_uri HAVING COUNT(*)>1)').fetchone()[0]
    subjects=db.execute('SELECT COUNT(DISTINCT subject) FROM facts').fetchone()[0]
    objects=db.execute('SELECT COUNT(DISTINCT object) FROM facts').fetchone()[0]
    entities=db.execute('SELECT COUNT(*) FROM (SELECT subject FROM facts UNION SELECT object FROM facts)').fetchone()[0]
    degrees=db.execute('SELECT CASE WHEN n=1 THEN \'one\' WHEN n<=8 THEN \'two_to_eight\' WHEN n<=32 THEN \'nine_to_32\' ELSE \'above_32\' END, COUNT(*) FROM (SELECT subject,COUNT(*) n FROM facts GROUP BY subject) GROUP BY 1').fetchall()
    db.close()
    if db_path.stat().st_size>reserve:raise ValueError('indexed source inventory exceeds reservation')
    result={'counts':dict(counts),'row_license_counts':dict(licenses),'source_dataset_counts':dict(datasets),
        'relation_counts':dict(relations),'subjects':subjects,'objects':objects,'entities':entities,
        'subject_degree_counts':dict(degrees),'duplicate_literal_triple_groups':duplicate_groups,
        'duplicate_source_assertion_uri_groups':duplicate_uris,'database_sha256':base.file_sha(db_path),
        'database_bytes':db_path.stat().st_size,'policy_sha256':base.file_sha(STATE/'policy.json'),
        'complete_source_stream_sha256':digest.hexdigest(),'accepted_training_rows':0,
        'model_requests':0,'semantic_task_diagnosis_and_exact_export_gates_still_required':True}
    base.save(STATE/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
