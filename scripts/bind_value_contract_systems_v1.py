"""Supplement exact query proofs with reviewed system and full row identities."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from barbet_inference.value_admission import bind_contract_identity
from barbet_inference.library import SkillLibrary
from barbet_inference.sft import SYSTEM
from scripts import publish_value_alignment_v1 as release

STATE=release.STATE


def trusted_systems():
    repo=release.REPO
    graph=SYSTEM+'\n\nAPPLICABLE SKILLS:\n'+SkillLibrary(repo).render(['grounded-evidence-answer'])[0]
    civic=SYSTEM+'\n\nAPPLICABLE SKILLS:\n'+(repo/'skills/taiwan-civic-values/SKILL.md').read_text()+'\n'+(repo/'skills/grounded-evidence-answer/SKILL.md').read_text()
    civic+='\n\n明示價值設定（價值不是外部事實證據）：\n'+'\n'.join(release.read(repo/'configs/taiwan-civic-policy-v1.json')['principles'])
    return {hashlib.sha256(text.encode()).hexdigest():{'profile':name,'text':text} for name,text in [('grounded_identifier_query',graph),('owner_civic_extract',civic)]}


def check(job):
    pa.set_cpu_count(1);t=pq.read_table(Path(job['root'])/job['path'],columns=['record_id','query_hash','messages_json','tools_json'])
    t=t.filter(pc.is_in(t['record_id'],value_set=pa.array(job['ids'])));result=[];trusted=trusted_systems()
    for row in t.to_pylist():
        try:digest=bind_contract_identity(row,trusted);status='BOUND';reason=None
        except Exception as exc:digest=None;status='UNBOUND';reason=str(exc)
        result.append({'record_id':row['record_id'],'query_hash':row['query_hash'],'system_sha256':digest,'status':status,'reason':reason})
    return result


def main():
    if (STATE/'system-binding-summary.json').exists():raise ValueError('First bindings already frozen')
    trusted=trusted_systems();release.save(STATE/'trusted-system-profiles.json',{'profiles':trusted,'driver_sha256':release.sha(Path(__file__)),'identity_validator_sha256':release.sha(release.REPO/'barbet_inference/value_admission.py')})
    proven={r['record_id'] for r in pq.read_table(STATE/'contract-dispositions.parquet').to_pylist() if r['status']=='PROVEN'}
    grouped=defaultdict(list)
    for row in pq.read_table(STATE/'tool-registry-scan-v1/candidates.parquet').to_pylist():
        if row['record_id'] in proven:grouped[(row['root'],row['path'])].append(row['record_id'])
    # Include the 96 civic articles outside the lexical candidate inventory.
    for p in sorted((release.prior.DEST/'data/wave-0011/records').glob('*.parquet')):
        key=(str(release.prior.DEST),str(p.relative_to(release.prior.DEST)))
        grouped[key]=list(set(grouped[key])|{r for r in pq.read_table(p,columns=['record_id'])['record_id'].to_pylist() if r in proven})
    jobs=[{'root':root,'path':path,'ids':ids} for (root,path),ids in grouped.items() if ids];results=[]
    with ProcessPoolExecutor(8) as pool:
        for n,rows in enumerate(pool.map(check,jobs),1):
            results.extend(rows)
            if n%200==0:print(json.dumps({'files':n,'bound_records_checked':len(results)}),flush=True)
    if {r['record_id'] for r in results}!=proven:raise ValueError('Contract identity coverage differs')
    pq.write_table(pa.Table.from_pylist(results),STATE/'system-bindings.parquet',compression='zstd')
    summary={'rows':len(results),'counts':dict(Counter(r['status'] for r in results)),'system_profiles':{key:value['profile'] for key,value in trusted.items()},'proofs_sha256':release.sha(STATE/'system-bindings.parquet'),'all_complete_record_identities_checked':True,'scope':'Additional required system-profile and full messages/tools identity gate; does not replace first contract results'}
    release.save(STATE/'system-binding-summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
