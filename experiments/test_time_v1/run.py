#!/usr/bin/env python3
"""Append-only batched inference. This process never opens an answer file."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from testtime.contracts import PublicRequest
from testtime.harness import Episode, Policy
from testtime.retrieval import Document


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def runtime_hashes():
    paths = sorted((ROOT / 'testtime').glob('*.py'))
    paths += [ROOT / 'testtime/cards.json', Path(__file__), HERE / 'models.json']
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mini', 'qwen'], required=True)
    parser.add_argument('--split', choices=['dev', 'validation', 'test'], required=True)
    parser.add_argument('--policies', nargs='+', required=True)
    parser.add_argument('--config', type=Path, default=HERE / 'policies.json')
    parser.add_argument('--work', type=Path, default=HERE / 'work')
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-model-len', type=int, default=49152)
    parser.add_argument('--max-seqs', type=int, default=64)
    parser.add_argument('--contracts', nargs='*')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--run-tag', default='main')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError('invalid shard')
    if args.limit and args.split != 'dev':
        raise ValueError('partial official validation/test is forbidden')
    if not args.run_tag.replace('-', '').replace('_', '').isalnum():
        raise ValueError('unsafe run tag')
    config = json.loads(args.config.read_text())
    specs = {name: config['policies'][name] for name in args.policies}
    policies = [Policy(name=name, **value) for name, value in specs.items()]
    hashes = runtime_hashes()
    model = json.loads((HERE / 'models.json').read_text())[args.model]
    input_path = args.work / f'{args.split}.inputs.jsonl'
    corpus_path = args.work / 'corpus.jsonl'
    input_hash, corpus_hash = sha(input_path.read_bytes()), sha(corpus_path.read_bytes())
    seal = None
    if args.split == 'test':
        seal = json.loads((HERE / 'FINAL_SELECTION.json').read_text())
        if seal['runtime_hashes'] != hashes or seal['input_sha256'] != input_hash or seal['corpus_sha256'] != corpus_hash:
            raise RuntimeError('test seal does not match inputs/runtime')
        if args.seed not in seal['seeds']:
            raise RuntimeError('unregistered final seed')
        for name, value in specs.items():
            if seal['policies'].get(name) != value or [args.model, name] not in seal['comparisons']:
                raise RuntimeError('policy/model pair is not in final selection')
        if args.contracts or args.limit:
            raise RuntimeError('final primary test may not be filtered')
    requests = [PublicRequest.from_dict(json.loads(x)) for x in input_path.read_text().splitlines()]
    if args.contracts:
        requests = [r for r in requests if r.output_contract in args.contracts]
    requests = [r for i, r in enumerate(requests) if i % args.shards == args.shard]
    if args.limit:
        requests = requests[:args.limit]
    if len({r.id for r in requests}) != len(requests):
        raise ValueError('duplicate request id')
    documents = {d['id']: Document(**d) for d in map(json.loads, corpus_path.read_text().splitlines())}
    cards = {c['id']: c for c in json.loads((ROOT / 'testtime/cards.json').read_text())['cards']}
    from transformers import AutoTokenizer
    import torch
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model['path'], local_files_only=True)
    model_config = json.loads((Path(model['path']) / 'config.json').read_text())
    eos = model_config.get('eos_token_id', model_config.get('text_config', {}).get('eos_token_id', tokenizer.eos_token_id))
    eos = sorted(set(x for x in (eos if isinstance(eos, list) else [eos]) if x is not None))
    terminal = [tokenizer.decode([i], skip_special_tokens=False) for i in eos]
    packages = {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'vllm']}
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    base = {'model': model, 'family': args.model, 'split': args.split, 'seed': args.seed,
        'input_sha256': input_hash, 'corpus_sha256': corpus_hash, 'runtime_hashes': hashes,
        'packages': packages, 'cuda_runtime': torch.version.cuda,
        'device_visibility': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'max_model_len': args.max_model_len, 'batch_size': args.batch_size, 'max_seqs': args.max_seqs,
        'eager': args.eager, 'shard': args.shard, 'shards': args.shards,
        'contracts': args.contracts, 'limit': args.limit, 'run_tag': args.run_tag,
        'eos_ids': eos, 'terminal_strings': terminal,
        'test_seal_sha256': sha(canonical(seal)) if seal else None}
    jobs = []
    for policy in policies:
        out = args.work / 'runs' / args.run_tag / args.split / args.model / policy.name / f'{args.seed}-s{args.shard}.jsonl'
        out.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**base, 'policy': as_policy(policy), 'request_ids': [r.id for r in requests]}
        fingerprint = sha(canonical(metadata))
        meta_path = out.with_suffix('.runtime.json')
        if meta_path.exists():
            if json.loads(meta_path.read_text())['run_fingerprint'] != fingerprint:
                raise RuntimeError('refusing changed-config resume: ' + str(out))
        else:
            if out.exists() and out.stat().st_size:
                raise RuntimeError('raw output missing provenance')
            meta_path.write_text(json.dumps({**metadata, 'run_fingerprint': fingerprint,
                'git_sha': git_sha, 'git_dirty': dirty, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()},
                ensure_ascii=False, indent=2) + '\n')
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                row = json.loads(line)
                if row['run_fingerprint'] != fingerprint or row['id'] in done:
                    raise RuntimeError('corrupt/duplicate output')
                done.add(row['id'])
        if not done <= {r.id for r in requests}:
            raise RuntimeError('unexpected completed request')
        jobs.append((policy, out, [r for r in requests if r.id not in done], fingerprint))
    if not any(j[2] for j in jobs):
        print(json.dumps({'event': 'already_complete'}), flush=True)
        return
    from vllm import LLM, SamplingParams
    kwargs = dict(model=model['path'], dtype='bfloat16', max_model_len=args.max_model_len,
        gpu_memory_utilization=.70, tensor_parallel_size=1, enforce_eager=args.eager,
        seed=args.seed, enable_prefix_caching=True, max_num_seqs=args.max_seqs,
        max_num_batched_tokens=16384, generation_config='vllm')
    if args.model == 'qwen':
        kwargs['limit_mm_per_prompt'] = {'image': 0, 'video': 0}
    loaded_at = time.monotonic()
    engine = LLM(**kwargs)
    load_seconds = time.monotonic() - loaded_at
    print(json.dumps({'event': 'loaded', 'model': args.model, 'seconds': load_seconds}), flush=True)
    for policy, out, pending, fingerprint in jobs:
        started = time.monotonic()
        with out.open('a') as stream:
            for offset in range(0, len(pending), args.batch_size):
                episodes = [Episode(r, policy, documents, cards, tokenizer, args.model, args.seed)
                            for r in pending[offset:offset + args.batch_size]]
                while any(not e.done for e in episodes):
                    active = [e for e in episodes if not e.done]
                    rendered = [e.generate_request() for e in active]
                    if any(len(ids) + sample['max_tokens'] > args.max_model_len for ids, sample, _ in rendered):
                        raise RuntimeError('context length exceeds engine ceiling')
                    batch_started = time.monotonic()
                    outputs = engine.generate([{'prompt_token_ids': ids} for ids, _, _ in rendered],
                        [SamplingParams(**sample, stop_token_ids=eos, skip_special_tokens=False)
                         for _, sample, _ in rendered], use_tqdm=False)
                    seconds = time.monotonic() - batch_started
                    for episode, output, (ids, _, _) in zip(active, outputs, rendered):
                        generated = output.outputs[0]
                        episode.accept(generated.text, prompt_tokens=len(ids), output_tokens=len(generated.token_ids),
                            finish_reason=generated.finish_reason, batch_seconds=seconds,
                            batch_size=len(active), terminal_strings=terminal)
                for episode in episodes:
                    row = {**episode.result(), 'model': args.model, 'run_fingerprint': fingerprint}
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                print(json.dumps({'event': 'progress', 'model': args.model, 'policy': policy.name,
                    'written': min(offset + args.batch_size, len(pending)), 'pending_at_start': len(pending),
                    'elapsed_seconds': time.monotonic() - started}), flush=True)
        out.with_suffix('.completed.json').write_text(json.dumps({'new_items': len(pending),
            'wall_seconds': time.monotonic() - started, 'engine_load_seconds': load_seconds,
            'raw_sha256': sha(out.read_bytes()), 'run_fingerprint': fingerprint,
            'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2) + '\n')


def as_policy(policy):
    from dataclasses import asdict
    return asdict(policy)


if __name__ == '__main__':
    main()
