"""Isolated Python execution used for public feedback and separate final scoring.

Linux/libseccomp required. Fail closed when isolation is unavailable. The child
preloads permitted modules, drops root, limits resources, and denies syscalls
outside a small allow-list before executing model code. No filesystem/network
tools, hidden answers, or ambient credentials are supplied to the worker.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run_python(code: str, tests: list[str] | None = None, setup: str = '',
               timeout: float = 5, want_answer: bool = False):
    checks = json.dumps({'tests': tests or [], 'setup': setup}, sort_keys=True)
    receipt = {'executor': 'isolated_python_v1',
        'artifact_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'checks_sha256': hashlib.sha256(checks.encode()).hexdigest(),
        'passed': False, 'status': 'not_executed', 'answer': None}
    if len(code) > 100_000 or len(checks) > 100_000:
        return {**receipt, 'status': 'input_limit', 'error': 'execution input too large'}
    started = time.monotonic()
    try:
        process = subprocess.run([sys.executable, '-I', str(Path(__file__).with_name('_worker.py'))],
            input=json.dumps({'code': code, 'tests': tests or [], 'setup': setup, 'want_answer': want_answer}),
            text=True, capture_output=True, timeout=timeout, cwd='/tmp',
            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}, close_fds=True)
        receipt['process_exit_code'] = process.returncode
        if process.returncode != 0:
            receipt.update(status='worker_failed', error=process.stderr[-1500:])
        else:
            result = json.loads(process.stdout)
            for key in ('passed', 'status', 'error', 'answer', 'stdout', 'checks_run'):
                if key in result:
                    receipt[key] = result[key]
    except subprocess.TimeoutExpired:
        receipt.update(status='timeout', error='execution exceeded wall-clock limit')
    except (OSError, ValueError) as exc:
        receipt.update(status='worker_failed', error=str(exc)[:1500])
    receipt['seconds'] = time.monotonic() - started
    return receipt
