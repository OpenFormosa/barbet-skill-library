"""Anonymous full-byte verification with a bounded transfer and fresh retries."""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import re
import subprocess
import tempfile
import time
from urllib.parse import quote

from .publication import REPO


class TransferError(RuntimeError):
    def __init__(self, code, retry_after=None):
        super().__init__(f'Anonymous curl transfer failed (exit {code})')
        self.retry_after = retry_after


def _retry_after(headers):
    blocks = re.split(r'\r?\n\r?\n', headers.strip())
    last = blocks[-1] if blocks else ''
    match = re.search(r'^retry-after:\s*(.+)$', last, re.I | re.M)
    if not match:
        return None
    value = match.group(1).strip()
    if value.isdigit():
        return int(value)
    try:
        return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
    except (ValueError, TypeError):
        return None


def _download_once(url, expected_bytes):
    # Fresh temporary body per attempt: no partial download can prefix a retry.
    deadline = max(180, min(600, 60 + expected_bytes / (1024 * 1024)))
    with tempfile.TemporaryDirectory(prefix='barbet-anonymous-readback-') as directory:
        header_path = directory + '/headers'
        with tempfile.TemporaryFile(dir=directory) as body:
            command = ['curl', '-q', '--fail', '--location', '--silent', '--show-error',
                       '--proto', '=https', '--proto-redir', '=https',
                       '--connect-timeout', '20', '--max-time', str(deadline),
                       '--speed-time', '45', '--speed-limit', '16384',
                       '--dump-header', header_path, '--output', '-', url]
            try:
                result = subprocess.run(command, stdout=body, stderr=subprocess.PIPE,
                                        timeout=deadline + 10, check=False)
            except subprocess.TimeoutExpired:
                raise TransferError('outer-timeout') from None
            if result.returncode:
                try:
                    with open(header_path) as handle:
                        delay = _retry_after(handle.read())
                except FileNotFoundError:
                    delay = None
                # Do not disclose signed redirect URLs or opaque response headers.
                raise TransferError(result.returncode, delay)
            body.seek(0)
            size = 0
            digest = hashlib.sha256()
            while chunk := body.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
            return size, digest.hexdigest()


def verify_remote(item, commit):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Readback requires an immutable Hub revision')
    remote = item['remote']
    if remote.startswith('/') or any(part in {'.', '..'} for part in remote.split('/')):
        raise ValueError('Invalid public repository path')
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{commit}/' + quote(remote, safe='/')
    for attempt in range(3):
        try:
            size, digest = _download_once(url, item['bytes'])
        except TransferError as exc:
            if attempt == 2:
                raise
            delay = exc.retry_after if exc.retry_after is not None else 2 + attempt
            # Do not retry sooner than an explicit service rate limit.
            if delay > 600:
                raise
            time.sleep(delay)
            continue
        if size != item['bytes'] or digest != item['sha256']:
            raise ValueError('Anonymous published byte hash differs')
        return {'path': remote, 'sha256': digest, 'bytes': size, 'anonymous': True,
                'transport_attempts': attempt + 1, 'transport': 'curl-bounded-v1'}
    raise AssertionError('Unreachable')
