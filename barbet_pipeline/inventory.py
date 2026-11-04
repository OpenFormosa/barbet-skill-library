"""Read-only inventory; unavailable release identities stay unknown."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import platform
import shutil
import subprocess

from .common import read_json, save_new, sha_file


def inventory(repo, work, output):
    repo, work, output = Path(repo), Path(work), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hashes = []
    paths = [repo / "configs/barbet_training_data_v2.json", repo / "configs/memory_sft_v2_v26_selection.json",
             work / "reports/value-alignment-v1/summary.json", work / "reports/value-alignment-v1/publication-manifest.json",
             work / "reports/value-alignment-v1/publication-final-commit.json"]
    for p in paths:
        hashes.append({"path": str(p), "sha256": sha_file(p), "bytes": p.stat().st_size})
    cache = Path('/root/.cache/huggingface/hub/models--voidful--barbet-skill/snapshots/67131067ff1825179d5e38e91d499159669e9c66')
    policy = read_json(repo / "configs/memory_sft_v2_v26_selection.json")
    v26 = {}
    for name, key in [("model.safetensors", "weights_sha256"), ("tokenizer.json", "tokenizer_sha256"),
                      ("chat_template.jinja", "native_template_sha256"), ("modeling_barbet.py", "modeling_sha256"),
                      ("load_barbet.py", "loader_sha256")]:
        actual = sha_file(cache / name)
        if actual != policy[key]:
            raise ValueError("Frozen v26 file changed: " + name)
        v26[name] = {"sha256": actual, "bytes": (cache / name).stat().st_size}
    configs = []
    legacy = Path('/root/Barbet/barbet-1b-base')
    candidates = [legacy] + sorted(Path('/root/.cache/huggingface/hub/models--OpenFormosa--barbet-1b-base/snapshots').glob('*'))
    for p in candidates:
        if not (p / 'config.json').exists():
            continue
        c = read_json(p / 'config.json')
        configs.append({"path": str(p), "config_sha256": sha_file(p / 'config.json'),
                        "num_hidden_layers": c.get('num_hidden_layers'), "head_dim": c.get('head_dim'),
                        "max_position_embeddings": c.get('max_position_embeddings'),
                        "weights_present": (p / 'model.safetensors').exists(),
                        "weights_sha256": sha_file(p / 'model.safetensors') if (p / 'model.safetensors').exists() else None,
                        "use_as_training_parent": False})
    remote = []
    from huggingface_hub import HfApi
    for name, kind in [("OpenFormosa/barbet-long-context-sft", "dataset"), ("OpenFormosa/barbet-1b-base", "model"), ("voidful/barbet-skill", "model")]:
        try:
            r = HfApi().repo_info(name, repo_type=kind)
            remote.append({"repo": name, "revision": r.sha, "status": "resolved_with_configured_credentials"})
        except Exception as exc:
            response = getattr(exc, 'response', None)
            remote.append({"repo": name, "revision": None, "status": "unavailable", "error_type": type(exc).__name__,
                           "http_status": getattr(response, 'status_code', None)})
    manifest = {"repository_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                "input_files": hashes, "v26_reference": {"revision": policy['revision'], "path": str(cache), "files": v26},
                "base_candidates": configs, "remote": remote,
                "training_parent": None, "training_parent_status": "BLOCKED_PENDING_RELEASE_WEIGHT_LINEAGE_VALIDATION",
                "remote_516_predictions": "NOT_AVAILABLE_IN_THIS_WORKSPACE", "training_started": False,
                "environment": {"python": platform.python_version(), "slurm_sbatch": shutil.which('sbatch'),
                                "disk_free_bytes": shutil.disk_usage(work).free,
                                "packages": {n: importlib.metadata.version(n) for n in ['pyarrow', 'huggingface-hub', 'tokenizers', 'jinja2', 'jsonschema']}},
                "scope": "CPU identity inventory, not model/backend/gradient capability verification"}
    save_new(output / 'inventory.json', manifest)
    return manifest
