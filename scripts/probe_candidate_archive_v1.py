"""Compare old and prospective storage on complete real source records, on CPU."""
import json
from pathlib import Path
from tokenizers import Tokenizer
from barbet_inference import candidate_archive as archive
from barbet_inference.answer_selection import NativeAnswerRenderer
from barbet_inference.memory import canonical
from scripts import build_memory_sft_v2 as base
from scripts.prepare_v26_answer_candidates import canonicalize, load_candidate

STATE = base.WORK / 'reports/memory-harness-v2-candidate-archive-v1'
SOURCE = base.WORK / 'reports/memory-harness-v2-ifm-v1'
MODEL = Path('/root/.cache/huggingface/hub/models--voidful--barbet-skill/snapshots/67131067ff1825179d5e38e91d499159669e9c66')


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'plan.json').exists():
        raise ValueError('archive source probe already started')
    files = sorted((SOURCE / 'normalized').glob('pilot-*.jsonl'))
    files += [sorted((SOURCE / 'math-full-v1/normalized').glob('full-*.jsonl'))[0]]
    plan = {'driver_sha256': base.file_sha(Path(__file__)),
        'archive_reader_sha256': base.file_sha(Path(archive.__file__)),
        'canonicalizer_sha256': base.file_sha(base.REPO / 'scripts/prepare_v26_answer_candidates.py'),
        'files': [{'path': str(p), 'sha256': base.file_sha(p), 'maximum_records': 32 if p.name.startswith('full-') else None} for p in files],
        'native_tokenizer_sha256': base.file_sha(MODEL / 'tokenizer.json'),
        'native_template_sha256': base.file_sha(MODEL / 'chat_template.jinja'),
        'native_renderer_sha256': base.file_sha(base.REPO / 'barbet_inference/answer_selection.py'),
        'model_requests': 0, 'accepted_training_rows': 0, 'all_source_programs_inert': True}
    policy = json.loads((base.REPO / 'configs/memory_sft_v2_v26_selection.json').read_text())
    if plan['native_tokenizer_sha256'] != policy['tokenizer_sha256'] or plan['native_template_sha256'] != policy['native_template_sha256']:
        raise ValueError('native rendering artifacts differ')
    base.save(STATE / 'plan.json', plan)
    renderer = NativeAnswerRenderer((MODEL / 'chat_template.jinja').read_text(), Tokenizer.from_file(str(MODEL / 'tokenizer.json')))
    pointers = []; raw_total = 0
    archive_path = STATE / 'source-records.bca'
    with archive.ArchiveWriter(archive_path) as writer:
        for file_id, spec in enumerate(plan['files']):
            with Path(spec['path']).open('rb') as handle:
                index = 0
                while True:
                    offset = handle.tell(); raw = handle.readline()
                    if not raw or spec['maximum_records'] is not None and index >= spec['maximum_records']:
                        break
                    index += 1; row = canonicalize(json.loads(raw))
                    compressed_offset, compressed_length = writer.append(raw); raw_total += len(raw)
                    identity = {'record_id': row['record_id'], 'candidate_identity': row['candidate_identity'], 'input_hash': row['ppl_input_hash']}
                    pointers.append({'original': {**identity, 'file_id': file_id, 'offset': offset, 'length': len(raw)},
                                     'archive': {**identity, 'file_id': 0, 'offset': compressed_offset, 'length': compressed_length}})
    receipts = []
    for pointer in pointers:
        original = load_candidate(pointer['original'], plan['files'])
        recovered = archive.load_candidate(pointer['archive'], [{'path': str(archive_path), 'format': archive.FORMAT}])
        if original != recovered:
            raise ValueError('complete source record differs after compression')
        a = renderer.render(original['messages'], original['tools'])
        b = renderer.render(recovered['messages'], recovered['tools'])
        if a != b:
            raise ValueError('native full text, tokens, masks or identities differ')
        receipts.append({'record_id': original['record_id'], 'domain': original['domain'],
                         'native_text_sha256': a['text_sha256'], 'sequence_tokens': a['sequence_tokens'],
                         'answer_tokens': a['answer_tokens'], 'all_record_fields_and_render_arrays_exact': True})
    result = {'samples': len(receipts), 'raw_payload_bytes': raw_total, 'archive_bytes': archive_path.stat().st_size,
        'archive_sha256': base.file_sha(archive_path), 'compression_ratio': archive_path.stat().st_size / raw_total,
        'plan_sha256': base.file_sha(STATE / 'plan.json'), 'all_comparisons_passed': True,
        'native_sequence_tokens_compared': sum(r['sequence_tokens'] for r in receipts),
        'native_supervised_tokens_compared': sum(r['answer_tokens'] for r in receipts),
        'samples_detail': receipts, 'model_requests': 0, 'accepted_training_rows': 0,
        'scope': 'Storage and full native rendering equivalence on these original first-diagnostic and 32 additional source records only. No new score, semantic review, training or old-cohort rewrite.'}
    base.save(STATE / 'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'samples_detail'}), flush=True)


if __name__ == '__main__':
    main()
