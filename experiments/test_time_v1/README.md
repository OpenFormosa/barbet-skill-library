# Frozen-model test-time study

The complete preregistered goal is [goal.json](goal.json); the readable plan and
current evidence are rendered in [the HTML report](../../docs/test-time-study.html).
This experiment makes **no weight updates**. Development and validation are in
progress; the report distinguishes these from the sealed final evaluation.

The deployment boundary accepts only a question, visible sources, tool schemas,
public examples and resource limits. Reference answers, hidden tests, subject
labels and source-model answers belong exclusively to the evaluator. Public
execution feedback is permitted; a successful process or matching quote is not
by itself a correct task answer.

The existing October 26 study is historical evidence, not this study's test set.
All numerical margins and search budgets in goal.json are engineering choices.
Changes after launch must be appended to amendments.json with a reason and the
affected run IDs; do not rewrite the registered goal to fit observed results.

Dataset downloads, test labels and large raw trajectories stay local under
`work/` (ignored by git). Public reports contain aggregate measurements, hashes,
source/version references and authored diagnostic fixtures. Dataset licenses
remain separate from the repository's Apache-2.0 license.

## Reproduction

Use Python 3.12 on Linux with `libseccomp2`. Run the repository checks with
`python scripts/build.py --check` and `python scripts/build_inference_cards.py --check`.
The inference environment used here is vLLM 0.23.0, transformers 5.12.1 and
PyTorch 2.11.0+cu130; exact per-run provenance records installed versions and
GPU visibility. Data preparation additionally uses huggingface-hub, pyarrow
and scikit-learn; paired confidence intervals use NumPy.

Download the two exact model revisions in `models.json`, set their local paths,
and verify against `model_integrity.json`. Run:

```bash
python experiments/test_time_v1/prepare_data.py
CUDA_VISIBLE_DEVICES=0 python experiments/test_time_v1/run.py --model mini --split dev --policies bare routed --eager
python experiments/test_time_v1/score.py --split dev
```

The default data-preparation mode replays `selected_ids.json` and verifies all
six original input/label SHA-256 values plus the corpus hash. It does not require
the earlier private workspace. `--original-selection` reproduces the initial
exclusion search only when those archived earlier-study inputs are available.
The replay was independently regenerated into a fresh directory before the
final test; all seven content hashes matched.

`run.py` reads public inputs and the passage corpus, not label files. `score.py`
is a separate process. `--split test` additionally requires an unchanged
`FINAL_SELECTION.json`; `seal.py` refuses to overwrite that selection or seal
after test trajectories exist. Keep failed and unpromoted configurations:
their costs and failure modes are evidence too.

The context probe is a separate **development diagnostic**. It uses evaluator
metadata to construct oracle-source and source-absent controls, then runs the
ordinary public-input harness. Oracle results are never deployment results.
The 32 source questions and the source/distractor sets are held constant across
positions; per-model tokenizer lengths are recorded. Condition IDs lead to
distinct random streams, so position effects include sampling variation.

Raw local artifacts are under `work/runs/`, `work/scores/`, `work/logs/` and
`work/context_probe/`. Reports expose question-level booleans and costs without
redistributing the passage corpus or hidden benchmark checks.
