"""Wait for finite inference jobs, finalize winners, then run their first audit.

This queue never publishes. ROOT inspects the first audit before continuing the
already-authorized compilation/publication; this is not a user-approval gate.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime,timezone
from scripts.prepare_v26_answer_candidates import STATE
from scripts.build_memory_sft_v2 import REPO,WORK,save


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt=STATE/'after-scoring-queue-v2.json'
    if receipt.exists():raise ValueError('queue already launched; inspect its PID and receipts')
    steps=[['scripts/finalize_v26_answer_selection.py'],['scripts/build_v26_selected_release.py','audit']]
    bindings={s[0]:sha(REPO/s[0]) for s in steps}
    plan={'pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),'steps':steps,'script_sha256':bindings,
        'waits_for':['eight complete canonical score buckets','complete strict episode identity guard'],
        'afterwards':'Inspect first winner-audit diagnosis, compile, package, compact and publish the authorized cumulative replacement.',
        'publishes_remotely':False,'model_training':False,'goal_complete':False}
    save(receipt,plan);print('waiting for PPL and episode identity receipts',flush=True)
    launch=json.loads((STATE/'score-launch-v2.json').read_text())
    while True:
        missing=[b for b in range(8) if not (STATE/f'scores/canonical-v2/bucket-{b}.summary.json').exists()]
        if not missing and (STATE/'episode-identity-v1/summary.json').exists():break
        for b in missing:
            pid=launch['workers'][b]['pid']
            try:command=Path(f'/proc/{pid}/cmdline').read_text()
            except FileNotFoundError:raise RuntimeError(f'scoring bucket {b} stopped without a completion receipt')
            if 'score_v26_answer_candidates.py' not in command:raise RuntimeError('worker PID identity changed')
        time.sleep(20)
    for n,step in enumerate(steps,1):
        if sha(REPO/step[0])!=bindings[step[0]]:raise RuntimeError('queued script changed; do not execute unbound replacement')
        if shutil.disk_usage(WORK).free<100*1024**3:raise RuntimeError('free disk below 100 GiB')
        log=WORK/'reports'/f'v26-selected-queue-v2-stage-{n}.log'
        print(json.dumps({'starting':step,'log':str(log)}),flush=True)
        with log.open('w') as f:
            result=subprocess.run([str(WORK/'.venv/bin/python'),*step],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        save(STATE/f'after-scoring-queue-v2-stage-{n}.json',{'step':step,'exit_code':result.returncode,'log':str(log),'completed_at':datetime.now(timezone.utc).isoformat()})
        if result.returncode:raise RuntimeError(f'queued stage {n} failed; preserve and inspect its log')
    print('Winner selection and first audit complete. Inspect diagnosis, then continue compilation and authorized publication.',flush=True)


if __name__=='__main__':main()
