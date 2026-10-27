#!/usr/bin/env python3
"""Build/install a wheel and check its CLI, skill assets and worker off-checkout."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='barbet-package-') as directory:
        temporary = Path(directory)
        subprocess.run([sys.executable, '-m', 'build', '--wheel', '--outdir', str(temporary / 'dist'), str(ROOT)], check=True)
        env = temporary / 'venv'
        venv.EnvBuilder(with_pip=True).create(env)
        python = env / 'bin/python'
        wheel, = (temporary / 'dist').glob('*.whl')
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', str(wheel)], check=True)
        process_env = dict(os.environ)
        process_env.pop('PYTHONPATH', None)
        expected = len(list((ROOT / 'skills').glob('*/skill.json')))
        code = '''
import json, subprocess, sys
from pathlib import Path
import barbet_inference
from barbet_inference import SkillLibrary, recommended_policy
from testtime.execution import run_python
assert not Path(barbet_inference.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
skills = SkillLibrary().list()
assert len(skills) == int(sys.argv[2])
assert recommended_policy().name == 'literal'
output = subprocess.check_output([str(Path(sys.executable).with_name('barbet-infer')), 'list-skills'], text=True)
assert len(output.splitlines()) == len(skills)
receipt = run_python('def double(x):\\n    return x * 2', ['assert double(4) == 8'])
assert receipt['passed'], receipt
print('INSTALLED_WHEEL_VALID', len(skills), 'skills; CLI, bundled policy and isolated worker')
'''
        subprocess.run([str(python), '-c', code, str(ROOT), str(expected)], cwd=temporary,
                       env=process_env, check=True)
        # Exercise the lightweight extra in a clean environment. A tiny authored
        # tokenizer fixture catches missing template dependencies without model
        # downloads, GPU initialization, or relying on the developer's packages.
        subprocess.run([str(python), '-m', 'pip', 'install', str(wheel) + '[tokenizer]'], check=True)
        tokenizer_check = '''
import importlib.util, json, subprocess, sys
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
assert importlib.util.find_spec('torch') is None
model = Path('tiny-tokenizer')
tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1}, unk_token='[UNK]'))
tokenizer.pre_tokenizer = Whitespace()
fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', eos_token='[EOS]')
fast.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}assistant: "
fast.save_pretrained(model)
(model / 'config.json').write_text(json.dumps({'model_type': 'llama', 'eos_token_id': 1}))
subprocess.run([str(Path(sys.executable).with_name('barbet-infer')), 'prepare', '--family', 'mini',
    '--model', str(model), '--local-files-only', '--prompt', 'What is 2 + 3?', '--contract', 'number',
    '--output', 'prepared.jsonl'], check=True)
row = json.loads(Path('prepared.jsonl').read_text())
assert row['prompt_tokens'] > 0 and '2 + 3' in row['prompt']
assert row['inference']['backend']['backend'] == 'tokenizer_only'
print('TOKENIZER_EXTRA_VALID; real chat template, no model download, no torch or GPU')
'''
        subprocess.run([str(python), '-c', tokenizer_check], cwd=temporary,
                       env=process_env, check=True)


if __name__ == '__main__':
    main()
