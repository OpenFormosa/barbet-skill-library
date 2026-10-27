"""Wait for complete real v26 scores, then select and compile the NI wave."""
import fcntl
import json
import os
import subprocess
import time

from scripts import build_memory_sft_v2 as base
from scripts.rank_natural_instructions_v26_v1 import STATE as SCORE
from scripts.finalize_natural_instructions_v26_v1 import STATE


def main():
    lock = (SCORE / 'release-coordinator.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    folder = SCORE / 'scores/canonical-v2'; paths = [folder / f'bucket-{b}.summary.json' for b in range(8)]
    while not all(p.exists() for p in paths): time.sleep(5)
    for p in paths:
        summary = json.loads(p.read_text())
        if summary['status'] != 'BUCKET_COMPLETE' or summary['limited_pilot']:
            raise ValueError('uncompleted or limited scoring bucket')
    for phase, marker in [('finalize', 'release-scope.json'), ('compile', 'compile-summary.json')]:
        if (STATE / marker).exists(): continue
        log = base.WORK / 'reports' / f'natural-instructions-{phase}-v1.log'
        with log.open('w') as handle:
            command = ['/root/barbet-context-sft-data/.venv/bin/python', 'scripts/finalize_natural_instructions_v26_v1.py', phase, '--workers', '16']
            result = subprocess.run(command, cwd=base.REPO, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('NI ' + phase + ' stopped; inspect preserved artifacts')
        print(json.dumps({'phase_complete': phase, 'log': str(log)}), flush=True)
    print(json.dumps({'status': 'NI_SELECTED_AND_COMPILED_NOT_YET_PUBLISHED', 'still_needs_winner_diagnosis_and_packaging': True}), flush=True)


if __name__ == '__main__': main()
