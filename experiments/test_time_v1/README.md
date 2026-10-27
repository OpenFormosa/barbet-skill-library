# Frozen-model test-time study

The complete preregistered goal is [goal.json](goal.json); the readable plan and
current evidence are rendered in [the HTML report](../../docs/test-time-study.html).
This experiment makes **no weight updates**. Initial status: implementation;
there are no new model results yet.

The deployment boundary accepts only a question, visible sources, tool schemas,
public examples and resource limits. Reference answers, hidden tests, subject
labels and source-model answers belong exclusively to the evaluator. Public
execution feedback is permitted; a successful process or matching quote is not
by itself a correct task answer.

The existing October 26 study is historical evidence, not this study's test set.
All numerical margins and search budgets in goal.json are engineering choices.
Changes after launch must be appended to amendments.jsonl with a reason and the
affected run IDs; do not rewrite the registered goal to fit observed results.

Dataset downloads, test labels and large raw trajectories stay local under
`work/` (ignored by git). Public reports contain aggregate measurements, hashes,
source/version references and authored diagnostic fixtures. Dataset licenses
remain separate from the repository's Apache-2.0 license.
