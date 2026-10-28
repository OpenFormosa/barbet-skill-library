"""Prospective metadata supplement; frozen row-subset rewriter is unchanged."""
import argparse
import fcntl
from pathlib import Path
import subprocess

from scripts import publish_barbet_readiness_v1 as release
from scripts import audit_barbet_continuation_v1 as audit

ORIGINAL_METADATA = release.metadata


def supplemented(plan, files):
    if release.sha(Path(release.__file__)) != release.read(release.STATE / 'policy.json')['driver_sha256']:
        raise ValueError('Frozen row-subset publisher changed')
    check = release.read(audit.STATE / 'native-collated-forward-v4.json')
    if not check['all_checks_passed'] or check['gradient_updates'] or check['absolute_delta'] > 1e-4:
        raise ValueError('Native no-grad forward not verified')
    removed = {rid for job in plan['jobs'] for rid in job['removed_ids']}
    if removed.intersection(check['records']): raise ValueError('Forward smoke used a removed row')
    ORIGINAL_METADATA(plan, files)
    release.save(release.DEST / release.FOLDER / 'native-collated-forward.json', check)
    release.save(release.DEST / release.FOLDER / 'native-flash-padding-first-failure.json',
                 release.read(audit.STATE / 'native-collated-forward-v3-kernel-diagnostic.json'))
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=release.parent.base.REPO, text=True).strip()
    release.save(release.DEST / release.FOLDER / 'metadata-supplement-binding.json', {
        'metadata_driver_sha256': release.sha(Path(__file__)), 'metadata_library_commit': code,
        'native_forward_sha256': release.sha(audit.STATE / 'native-collated-forward-v4.json'),
        'first_failed_kernel_log_sha256': release.sha(audit.STATE.parent / 'v26-native-collated-forward-v3.log'),
        'frozen_payload_driver_sha256': release.sha(Path(release.__file__)),
        'all_row_values_and_answer_selection_unchanged_by_supplement': True})
    config_path = release.DEST / 'recipe/barbet-training-v2.json'
    config = release.read(config_path)
    config.update(attention_backend='default mask-capable SDPA kernels',
                  force_flash_only_with_padding_mask=False,
                  long_sequence_batching='Prefer unpadded microbatches or equal-length buckets; verify runtime before scaling',
                  native_short_collated_forward_verified=True,
                  retrieval_input_whitelist=['messages', 'tools', 'causally_available_memory_documents'],
                  exclude_audit_and_source_refs_from_prompt=True)
    release.save(config_path, config)
    paragraph = (f'\nThe actual frozen v26 model also passed a two-row no-grad padded forward: '
                 f'masked loss {check["native_masked_loss"]:.9f} exactly matched an independent explicit '
                 f'single-shift CE (absolute delta {check["absolute_delta"]:.9f}; '
                 f'{check["supervised_tokens"]} supervised tokens). This is a short wiring smoke, '
                 'not training, a gradient/long-sequence test or model improvement evidence. '
                 'Forcing FLASH_ATTENTION alone failed on the real padding mask; that first failure '
                 'is preserved. The successful smoke used default enabled SDPA kernels.\n')
    with (release.DEST / 'README.md').open('a') as handle: handle.write(paragraph)
    with (release.DEST / 'TRAINING.md').open('a') as handle:
        handle.write('''
Attention runtime: do not force Flash-only kernels for padded-mask batches. The
native short-batch smoke passed with default enabled mask-capable SDPA kernels.
For long sequences, prefer unpadded microbatches/equal-length buckets and verify
memory and kernel support in the actual trainer before scaling. No long-gradient
or cross-example Mamba-state-reset claim is made by this short no-grad smoke.

Memory/RAG input must use the intended messages, skills and causally available
documents. Do not index audit reports, source_refs_json, gold targets or source
construction traces as model-visible evidence. Dataset adapters exclude those
metadata fields; keep that boundary in the actual training/evaluation pipeline.
''')
    path = release.DEST / release.FOLDER / 'report.html'
    text = path.read_text(); end = text.index('<pre>')
    addition = (f'<p><strong>原生 v26 的短 batch forward 已實測。</strong>兩筆保留資料長度 '
                f'{check["lengths"]}，padding {check["right_padding_tokens"]} tokens，'
                f'監督 {check["supervised_tokens"]} tokens。模型 loss 與獨立 single-shift CE 均為 '
                f'{check["native_masked_loss"]:.9f}，差值 {check["absolute_delta"]:.9f}。'
                '全程 eval/no-grad，沒有更新參數。第一次強制 Flash-only 在 padding mask 上失敗，'
                '原始紀錄保留；改用預設 mask-capable SDPA 路徑後通過。'
                '此測試不等於長序列 gradient/packing 或能力提升驗收。</p>')
    path.write_text(text[:end] + addition + text[end:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['probe', 'publish'])
    args = parser.parse_args()
    release.metadata = supplemented
    if args.phase == 'probe':
        release.DEST = release.STATE / 'metadata-probe-v2'; release.DEST.mkdir(exist_ok=True)
        supplemented(release.read(release.STATE / 'plan.json'), [])
        print('Prospective metadata probe complete; native arrays unchanged')
    else:
        with (release.parent.base.WORK / 'reports/memory-harness-v2/publisher.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); release.publish()
