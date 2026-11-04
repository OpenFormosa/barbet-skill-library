import hashlib
from unittest.mock import patch
import pytest
from barbet_pipeline import anonymous_readback as mod

ITEM = {'remote':'full-v4/example.parquet','bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()}
COMMIT = 'a'*40


def test_success_is_full_byte_verification():
    with patch.object(mod, '_download_once', return_value=(3, ITEM['sha256'])) as call:
        proof = mod.verify_remote(ITEM, COMMIT)
    assert proof['anonymous'] and proof['bytes'] == 3
    assert proof['transport_attempts'] == 1
    assert call.call_args[0][0].endswith('/' + COMMIT + '/' + ITEM['remote'])


def test_retry_starts_fresh_and_respects_rate_limit():
    with patch.object(mod, '_download_once', side_effect=[mod.TransferError(22, 7), (3, ITEM['sha256'])]), patch.object(mod.time, 'sleep') as sleep:
        proof = mod.verify_remote(ITEM, COMMIT)
    sleep.assert_called_once_with(7)
    assert proof['transport_attempts'] == 2


@pytest.mark.parametrize('result', [(2, ITEM['sha256']), (3, '0'*64)])
def test_integrity_failure_is_not_retried(result):
    with patch.object(mod, '_download_once', return_value=result) as call:
        with pytest.raises(ValueError, match='byte hash'):
            mod.verify_remote(ITEM, COMMIT)
    assert call.call_count == 1


def test_retries_are_bounded():
    with patch.object(mod, '_download_once', side_effect=mod.TransferError(28)) as call, patch.object(mod.time, 'sleep'):
        with pytest.raises(mod.TransferError):
            mod.verify_remote(ITEM, COMMIT)
    assert call.call_count == 3


def test_long_retry_after_is_not_bypassed():
    with patch.object(mod, '_download_once', side_effect=mod.TransferError(22, 900)), patch.object(mod.time, 'sleep') as sleep:
        with pytest.raises(mod.TransferError):
            mod.verify_remote(ITEM, COMMIT)
    sleep.assert_not_called()


def test_final_redirect_response_controls_rate_limit():
    assert mod._retry_after('HTTP/2 302\r\nRetry-After: 1\r\n\r\nHTTP/2 429\r\nRetry-After: 37\r\n\r\n') == 37


def test_immutable_revision_required():
    with pytest.raises(ValueError, match='immutable'):
        mod.verify_remote(ITEM, 'main')
