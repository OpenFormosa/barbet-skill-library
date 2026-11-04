import hashlib
from pathlib import Path
import pytest
from barbet_pipeline import anonymous_readback_ranged as mod
from barbet_pipeline.full_publication_streamed import verify_files
from barbet_pipeline.common import read_json


def test_exact_range():
    mod._validate_range('HTTP/2 302\r\n\r\nHTTP/2 206\r\ncontent-range: bytes 3-5/8\r\n\r\n',3,5,8)


@pytest.mark.parametrize('headers',['HTTP/2 200\ncontent-range: bytes 3-5/8','HTTP/2 206\ncontent-range: bytes 2-5/8','HTTP/2 206\ncontent-range: bytes 3-5/9','HTTP/2 206\n'])
def test_invalid_range_rejected(headers):
    with pytest.raises(ValueError):mod._validate_range(headers,3,5,8)


def test_whole_hash_uses_original_byte_order(monkeypatch):
    content=b'abcdefghijk'
    monkeypatch.setattr(mod,'RANGE_THRESHOLD',1);monkeypatch.setattr(mod,'CHUNK_BYTES',3)
    def transfer(url,path,total,byte_range):
        start,end=byte_range;path.write_bytes(content[start:end+1]);return path
    monkeypatch.setattr(mod,'_transfer',transfer)
    assert mod._download_once('public-url',len(content))==(len(content),hashlib.sha256(content).hexdigest())


def test_completed_files_and_chunks_resume(tmp_path):
    files=[{'remote':f'file-{i}','bytes':i,'sha256':str(i)} for i in range(67)]
    def verify(item,commit):return {'path':item['remote'],'bytes':item['bytes'],'sha256':item['sha256'],'anonymous':True}
    result=verify_files(files,'a'*40,tmp_path,0,verify)
    assert len(result)==67 and len(list((tmp_path/'readback').glob('*.json')))==3
    def fail(*args):raise AssertionError('must reuse successful proof')
    assert verify_files(files,'a'*40,tmp_path,0,fail)==result
    changed=[dict(f) for f in files];changed[0]['sha256']='changed'
    with pytest.raises(ValueError):verify_files(changed,'a'*40,tmp_path,0,fail)
