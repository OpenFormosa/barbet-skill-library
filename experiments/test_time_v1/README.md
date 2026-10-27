# Frozen-model test-time study

The complete preregistered goal is [goal.json](goal.json); the readable plan and
current evidence are rendered in [the HTML report](../../docs/test-time-study.html).
This experiment makes **no weight updates**. The 18-policy search and all 30
sealed final jobs are complete: 646 held-out questions, three seeds, ten arms.
MiniCPM's selected harness improves the registered four-domain macro from
38.02% to 50.39% (+12.37 pp, paired 95% CI +9.90 to +14.97), while the same
harness with Qwen3.5-9B reaches 76.24%. This establishes a task-system gain,
not 9B equivalence. See `final_comparisons.json` and `final_diagnostics.json`.

The strict mathematics metric includes final-format adherence. A separately
labeled post-hoc readout diagnostic recovers already-emitted standalone final
numbers for every arm; it does not read unfinished thoughts or change the
sealed scorer. This diagnostic gives MiniCPM bare 72.92% and selected 71.09%
on mathematics, so the strict math gain is **not evidence of better numerical
reasoning**. `numeric_diagnostics.json` preserves that sensitivity analysis.

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

Final analysis can be regenerated after all sealed jobs complete:

```bash
python experiments/test_time_v1/score.py --split test --run-tag final
python experiments/test_time_v1/analyze_final.py
python experiments/test_time_v1/code_budget_analysis.py
python experiments/test_time_v1/numeric_diagnostics.py
python experiments/test_time_v1/collect_results.py
python experiments/test_time_v1/render_report.py
```

The analysis commands require NumPy. Code-budget analysis replays prefixes of
the exact same frozen trajectories; it makes no new model calls. Three calls
improve code accuracy over two, so stopping at the registered budget is not a
claim that further improvement is impossible. BF16 batch composition can
change sampled outputs even with the same seed; independent policy arms are
not bitwise paired generations. Reported costs include every returned call.
