"""Inventory actual pinned model files; never infer pretraining from repo name."""
from pathlib import Path
import json
import math
from collections import Counter
from barbet_pipeline.common import sha_file, save_new


def verify(folder, model_id, revision, output):
    from safetensors import safe_open
    from huggingface_hub import HfApi
    p = Path(folder)
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    assert info.sha == revision
    remote = {x.rfilename: x for x in info.siblings}
    files = {}
    for name in ['model.safetensors', 'config.json', 'tokenizer.json', 'chat_template.jinja',
                 'modeling_barbet.py', 'configuration_barbet.py', 'load_barbet.py',
                 'megatron_long_context.py', 'release_manifest.json']:
        if not (p / name).is_file():
            continue
        files[name] = {'sha256': sha_file(p/name), 'bytes': (p/name).stat().st_size}
        lfs = remote[name].lfs
        if lfs:
            assert files[name]['sha256'] == lfs.sha256
    tensors = {}
    census = Counter()
    with safe_open(str(p/'model.safetensors'), framework='numpy') as f:
        for key in f.keys():
            s = f.get_slice(key)
            shape, dtype = s.get_shape(), s.get_dtype()
            tensors[key] = {'shape': shape, 'dtype': dtype}
            census[dtype] += math.prod(shape)
    config = json.loads((p/'config.json').read_text())
    assert config['num_hidden_layers'] == 29 and config['max_position_embeddings'] == 1048576
    assert config['head_dim'] == 128
    assert files['tokenizer.json']['sha256'] == 'f84754e42cc4b08d5573a6b7306635e86ed04fff68ee707662d76d7044917176'
    release = json.loads((p/'release_manifest.json').read_text()) if (p/'release_manifest.json').exists() else {}
    receipt = {'model_id': model_id, 'revision': revision, 'files': files, 'config': config,
        'tensor_layout': tensors, 'dtype_element_census': dict(census),
        'release_declared_stage': release.get('stage'),
        'release_declared_parent': release.get('parent_revision'),
        'training_parent_selected': False, 'base_name_is_not_pretraining_evidence': True,
        'earlier_lineage': 'NOT_INDEPENDENTLY_VERIFIED', 'gpu_smoke': 'NOT_RUN'}
    save_new(output, receipt)
    return receipt


if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser()
    for n in ['folder', 'model-id', 'revision', 'output']:
        a.add_argument('--'+n, required=True)
    ns = a.parse_args()
    result = verify(ns.folder, ns.model_id, ns.revision, ns.output)
    print(json.dumps({k: result[k] for k in ['model_id', 'revision', 'dtype_element_census', 'release_declared_stage']}))
