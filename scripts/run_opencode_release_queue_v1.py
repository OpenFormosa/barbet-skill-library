"""Select and compile only after complete real scores and frozen prior aliases."""
import fcntl
import json
import os
import subprocess
import time

from scripts import build_memory_sft_v2 as base
from scripts.rank_opencode_v26_v1 import STATE as SCORE
from scripts.freeze_opencode_prior_v1 import STATE as FROZEN
from scripts.finalize_opencode_v26_v1 import STATE


if __name__ == '__main__':
    with (SCORE / 'release-coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = [SCORE / 'scores/canonical-v2' / f'bucket-{n}.summary.json' for n in range(8)]
        while not (FROZEN / 'summary.json').exists() or not all(p.exists() for p in paths): time.sleep(5)
        if any(json.loads(p.read_text())['status'] != 'BUCKET_COMPLETE' or json.loads(p.read_text())['limited_pilot'] for p in paths):
            raise ValueError('scoring incomplete')
        for phase, marker in [('finalize', 'release-scope.json'), ('compile', 'compile-summary.json')]:
            if (STATE / marker).exists(): continue
            log = base.WORK / 'reports' / f'opencode-{phase}-v1.log'
            with log.open('w') as handle:
                result = subprocess.run(['/root/barbet-context-sft-data/.venv/bin/python',
                    'scripts/finalize_opencode_v26_v1.py', phase], cwd=base.REPO, env=os.environ.copy(),
                    stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError('OpenCode ' + phase + ' failed; inspect preserved artifacts')
            print(json.dumps({'phase_complete': phase}), flush=True)
        print(json.dumps({'status': 'SELECTED_AND_COMPILED_NOT_YET_PUBLISHED',
                          'still_needs_first_winner_diagnosis_and_exact_public_package': True}), flush=True)
