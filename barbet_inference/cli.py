"""User-facing inference commands, independent of benchmark input/label files."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from importlib.resources import files
import json
import os
from pathlib import Path
import sys

from testtime.contracts import strict_json
from testtime.harness import Policy

from .backends import HTTPBackend, TokenizerBackend, VLLMBackend
from .library import SkillLibrary
from .runtime import CONTRACTS, Harness, make_request, recommended_policy, validate_policy, validate_request


def read_records(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.json':
        value = strict_json(text)
        return value if isinstance(value, list) else [value]
    return [strict_json(line) for line in text.splitlines() if line.strip()]


def parser():
    root = argparse.ArgumentParser(prog='barbet-infer', description='Skill-guided inference without weight updates')
    commands = root.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list-skills', help='List the bundled library; no model dependencies needed')
    listing.add_argument('--library', type=Path)
    listing.add_argument('--scope')
    for name in ('prepare', 'run'):
        cmd = commands.add_parser(name, help='Render prompts only' if name == 'prepare' else 'Generate answers with bounded feedback')
        cmd.add_argument('--family', choices=['mini', 'qwen'], required=True)
        cmd.add_argument('--model', help='HF model ID or local snapshot; defaults to the pinned family profile')
        cmd.add_argument('--revision', help='Model and tokenizer revision; custom model IDs are unpinned unless supplied')
        cmd.add_argument('--local-files-only', action='store_true')
        source = cmd.add_mutually_exclusive_group(required=True)
        source.add_argument('--prompt')
        source.add_argument('--request-file', type=Path, help='Public request JSON object/array or JSONL')
        cmd.add_argument('--contract', choices=CONTRACTS, help='Required with --prompt; JSON requests carry their own contract')
        cmd.add_argument('--documents', type=Path, help='Source documents as JSON/JSONL')
        cmd.add_argument('--policy', choices=['recommended', 'bare'], default='recommended')
        cmd.add_argument('--policy-file', type=Path, help='Custom Policy fields as JSON; overrides --policy')
        cmd.add_argument('--skill', action='append', default=[], help='Explicit library skill; repeat to compose, replacing automatic short cards')
        cmd.add_argument('--library', type=Path, help='Alternate repository or skills directory')
        cmd.add_argument('--no-execution', action='store_true')
        cmd.add_argument('--max-calls', type=int, choices=[1, 2, 3], default=3)
        cmd.add_argument('--input-tokens', type=int)
        cmd.add_argument('--max-output-tokens', type=int)
        cmd.add_argument('--continuation-input-tokens', type=int)
        cmd.add_argument('--max-model-len', type=int, default=49152)
        cmd.add_argument('--seed', type=int, default=20261027)
        cmd.add_argument('--output', type=Path, required=True, help='New JSONL result file; existing files are never overwritten')
        if name == 'run':
            cmd.add_argument('--backend', choices=['vllm', 'http'], default='vllm')
            cmd.add_argument('--base-url', help='HTTP backend URL including /v1')
            cmd.add_argument('--served-model', help='Remote server model name, if different from --model')
            cmd.add_argument('--api-key-env', default='BARBET_API_KEY')
            cmd.add_argument('--timeout', type=float, default=120)
            cmd.add_argument('--batch-size', type=int, default=8)
            cmd.add_argument('--gpu-memory-utilization', type=float, default=.7)
            cmd.add_argument('--tensor-parallel-size', type=int, default=1)
            cmd.add_argument('--cuda-graphs', action='store_true', help='Enable vLLM graph compilation; default uses eager mode')
            cmd.add_argument('--trace', type=Path, help='New JSONL log of every returned model call, before host checks')
            cmd.add_argument('--include-trace', action='store_true', help='Include complete trajectories in result records too')
    return root


def policy_from_args(args):
    if args.policy_file:
        value = strict_json(args.policy_file.read_text())
        if not isinstance(value, dict):
            raise ValueError('policy file must be an object of Policy fields')
        policy = Policy(**{'name': 'custom', **value})
    else:
        policy = recommended_policy() if args.policy == 'recommended' else Policy(name='bare')
    overrides = {k: dict(v) for k, v in policy.contract_overrides.items()}
    changes = {key: value for key, value in {
        'input_tokens': args.input_tokens, 'output_tokens': args.max_output_tokens,
        'continuation_input_tokens': args.continuation_input_tokens}.items() if value is not None}
    if args.no_execution:
        changes.update(execute=False, repair=0)
    for value in overrides.values():
        value.update(changes)
    policy = replace(policy, **changes, contract_overrides=overrides)
    validate_policy(policy, args.max_calls)
    return policy


def write_json(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def resolve_model_path(model, revision, local_files_only):
    if not local_files_only or Path(model).is_dir():
        return model
    # Some Hub versions require every repository file for snapshot_download,
    # including README files. Resolve the cached commit through config.json;
    # the tokenizer/engine then validate their actual required assets offline.
    from huggingface_hub import try_to_load_from_cache
    config = try_to_load_from_cache(model, 'config.json', revision=revision or 'main')
    if not isinstance(config, str) or not Path(config).is_file():
        raise ValueError('model config is not cached; provide a local --model snapshot or allow downloading')
    return str(Path(config).parent)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'list-skills':
            rows = SkillLibrary(args.library).list()
            for row in rows:
                if not args.scope or row['scope'] == args.scope:
                    write_json(sys.stdout, row)
            return 0
        if args.prompt is not None:
            if not args.contract:
                raise ValueError('--prompt requires --contract')
            requests = [make_request(args.prompt, args.contract)]
        else:
            if args.contract:
                raise ValueError('--contract is for --prompt; request files specify output_contract per record')
            requests = [validate_request(r) for r in read_records(args.request_file)]
        if not requests or len({r.id for r in requests}) != len(requests):
            raise ValueError('requests must be nonempty and have unique IDs')
        documents = read_records(args.documents) if args.documents else []
        policy = policy_from_args(args)
        library = SkillLibrary(args.library) if args.skill else None
        if library:
            library.render(args.skill)
        profile = strict_json(files('barbet_inference').joinpath('profiles.json').read_text())[args.family]
        model = args.model or profile['model']
        revision = args.revision if args.model else (args.revision or profile['revision'])
        model = resolve_model_path(model, revision, args.local_files_only)
        if args.command == 'run' and args.backend == 'http' and not args.base_url:
            raise ValueError('--backend http requires --base-url')
        targets = [args.output] + ([args.trace] if getattr(args, 'trace', None) else [])
        if len({p.resolve() for p in targets}) != len(targets):
            raise ValueError('output and trace paths must differ')
        if any(p.exists() for p in targets):
            raise ValueError('output/trace already exists; choose a new path')
        with ExitStack() as stack:
            for path in targets:
                path.parent.mkdir(parents=True, exist_ok=True)
            output = stack.enter_context(args.output.open('x', encoding='utf-8'))
            trace = stack.enter_context(args.trace.open('x', encoding='utf-8')) if getattr(args, 'trace', None) else None
            common = dict(revision=revision, max_model_len=args.max_model_len, local_files_only=args.local_files_only)
            # Preflight every prompt before allocating the local generation engine.
            with redirect_stdout(sys.stderr):
                tokenizer_backend = TokenizerBackend(model, **common)
                harness_args = dict(family=args.family, policy=policy, documents=documents,
                    skills=args.skill, library=library, max_calls=args.max_calls, seed=args.seed)
                harness = Harness(tokenizer_backend, **harness_args)
                prepared = [harness.prepare(r) for r in requests]
                if args.command == 'run':
                    common.update(tokenizer=tokenizer_backend.tokenizer, eos_ids=tokenizer_backend.eos_ids)
                    if args.backend == 'vllm':
                        backend = VLLMBackend(model, family=args.family, batch_size=args.batch_size,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            tensor_parallel_size=args.tensor_parallel_size, eager=not args.cuda_graphs, **common)
                    else:
                        backend = HTTPBackend(model, base_url=args.base_url,
                            served_model=args.served_model or args.model or profile['model'],
                            api_key=os.environ.get(args.api_key_env), timeout=args.timeout, **common)
                    harness = Harness(backend, **harness_args)
            if args.command == 'prepare':
                for row in prepared:
                    write_json(output, row)
            else:
                def log_call(value):
                    if trace:
                        write_json(trace, value)
                if trace:
                    write_json(trace, {'event': 'start', 'backend': backend.metadata,
                        'request_ids': [r.id for r in requests], 'seed': args.seed})
                with redirect_stdout(sys.stderr):
                    for row in harness.run_many(requests, on_call=log_call):
                        if not args.include_trace:
                            row.pop('trace')
                        write_json(output, row)
                        print(json.dumps({'id': row['id'], 'calls': row['calls'],
                            'status': row['validation']['status']}, ensure_ascii=False), file=sys.stderr)
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print('barbet-infer: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
