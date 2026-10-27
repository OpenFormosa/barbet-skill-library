"""Prospective indentation correction after the v1 pre-scan assertion failure."""
import json
from pathlib import Path
from barbet_inference import code_source_scope_v2
from scripts import audit_ifm_code_scope_v1 as audit
from scripts import build_memory_sft_v2 as base

STATE = audit.SELECTED / 'source-scope-v2'


def main():
    STATE.mkdir(exist_ok=True)
    old = audit.SELECTED / 'source-scope-v1'
    if (old / 'summary.json').exists() or (old / 'query-holds.json').exists():
        raise ValueError('v1 failure no longer before complete source scan')
    binding = {'driver_sha256': base.file_sha(Path(__file__)),
        'unchanged_scan_driver_sha256': base.file_sha(Path(audit.__file__)),
        'original_pre_scan_policy_sha256': base.file_sha(old / 'policy.json'),
        'original_failure_log_sha256': base.file_sha(base.WORK / 'reports/ifm-code-source-scope-v1.log'),
        'corrected_checker_sha256': base.file_sha(Path(code_source_scope_v2.__file__)),
        'reason': 'v1 checked indentation-bearing Markdown code without dedent; the illustrative-output guard stopped before the full scan. No source decision, model request, answer, or score was changed.',
        'test_receipt_sha256': base.file_sha(base.WORK / 'reports/memory-harness-v2/code-source-scope-tests-v2.log')}
    path = STATE / 'prospective-correction-v2.json'
    if path.exists() and json.loads(path.read_text()) != binding: raise ValueError('scope correction changed')
    if not path.exists(): base.save(path, binding)
    audit.STATE = STATE; audit.scope = code_source_scope_v2; audit.main()


if __name__ == '__main__': main()
