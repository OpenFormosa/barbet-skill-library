"""Bounded anonymous range downloads, with whole-file SHA-256 as final gate."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import re
import subprocess
import tempfile
import threading
import time
from urllib.parse import quote
from .anonymous_readback import TransferError, _retry_after
from .publication import REPO

CONNECTIONS = threading.BoundedSemaphore(8)
CHUNK_BYTES = 8 * 1024 * 1024
RANGE_THRESHOLD = 16 * 1024 * 1024


def _validate_range(headers, start, end, total):
    last = re.split(r'\r?\n\r?\n', headers.strip())[-1]
    status = re.match(r'HTTP/\S+\s+(\d+)', last)
    content = re.search(r'^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$', last, re.I | re.M)
    if not status or status.group(1) != '206' or not content or tuple(map(int, content.groups())) != (start,end,total):
        raise ValueError('Anonymous range response does not cover the requested bytes')


def _transfer(url, path, total, byte_range=None):
    header_path = str(path) + '.headers'
    command = ['curl','-q','--fail','--location','--silent','--show-error',
               '--proto','=https','--proto-redir','=https','--connect-timeout','20',
               '--max-time','180','--speed-time','45','--speed-limit','16384',
               '--dump-header',header_path,'--output',str(path)]
    if byte_range is not None:
        command += ['--range',f'{byte_range[0]}-{byte_range[1]}']
    command.append(url)
    try:
        with CONNECTIONS:
            result = subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=190,check=False)
    except subprocess.TimeoutExpired:
        raise TransferError('outer-timeout') from None
    headers = Path(header_path).read_text() if Path(header_path).exists() else ''
    if result.returncode:
        raise TransferError(result.returncode, _retry_after(headers))
    expected = total if byte_range is None else byte_range[1]-byte_range[0]+1
    if path.stat().st_size != expected:
        raise ValueError('Anonymous response byte length differs')
    if byte_range is not None:
        _validate_range(headers,*byte_range,total)
    return path


def _download_once(url, expected_bytes):
    with tempfile.TemporaryDirectory(prefix='barbet-readback-ranges-') as directory:
        directory = Path(directory)
        if expected_bytes <= RANGE_THRESHOLD:
            paths = [_transfer(url,directory/'full',expected_bytes)]
        else:
            spans = [(start,min(start+CHUNK_BYTES,expected_bytes)-1) for start in range(0,expected_bytes,CHUNK_BYTES)]
            def fetch(pair):
                index, span = pair
                return _transfer(url,directory/f'{index:06d}',expected_bytes,span)
            with ThreadPoolExecutor(4) as pool:
                paths = list(pool.map(fetch,enumerate(spans)))
        digest = hashlib.sha256();size = 0
        for path in paths:
            with path.open('rb') as handle:
                while chunk := handle.read(1024*1024):
                    size += len(chunk);digest.update(chunk)
        return size,digest.hexdigest()


def verify_remote(item, commit):
    if not re.fullmatch(r'[0-9a-f]{40}',commit):raise ValueError('Readback requires an immutable Hub revision')
    remote=item['remote']
    if remote.startswith('/') or any(p in {'.','..'} for p in remote.split('/')):raise ValueError('Invalid public repository path')
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{commit}/'+quote(remote,safe='/')
    for attempt in range(3):
        try:size,digest=_download_once(url,item['bytes'])
        except TransferError as error:
            delay=error.retry_after if error.retry_after is not None else 2+attempt
            if attempt==2 or delay>600:raise
            time.sleep(delay);continue
        if size!=item['bytes'] or digest!=item['sha256']:raise ValueError('Anonymous published byte hash differs')
        return {'path':remote,'sha256':digest,'bytes':size,'anonymous':True,
                'transport_attempts':attempt+1,'transport':'curl-ranges-v2','all_bytes_hashed_in_original_order':True}
    raise AssertionError('Unreachable')
