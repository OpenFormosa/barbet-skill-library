# Optional causal procedure router

`extension.select({"messages": [...], "tools": [...]})` suggests a minimal procedure after a more specific application router finds no applicable skill. It returns a skill name and input-only trigger evidence, or `None`. The registry itself does not prove a tool can fulfill the request: `tool-capability-triage` explicitly compares capabilities and permits clarification or a supported alternative.

This router uses English and Traditional Chinese lexical cues. It is a reproducible baseline, not a semantic classifier, safety policy, execution authorization or proof of answer correctness. A candidate must still satisfy the skill's Use When / Do Not Use When and the application's higher-priority instructions. No universal fallback is applied to noise, social turns or unsupported input roles. Do not infer domain correctness from a coverage percentage.

Only causal `messages` and `tools` are accepted. Do not pass target answers, later tool observations, source names, gold labels, splits or evaluation scores. Different answers to an identical input must receive the same routing decision. Keep the original task, tool schemas and response format; run answer compatibility checks separately and never use those checks to choose a route.

For SFT derivatives, pin the library commit and content hashes, keep inserted skill text out of supervised labels, preserve source `loss=false`, re-tokenize the entire sample without truncation, and recalculate selection budgets. Each skill block applies only to its next assistant prediction point. Local data consumers must preserve provenance, root grouping, splits and exclusions. This repository contains no training samples or target answers and does not publish a dataset update.

Run `python scripts/build.py --check` to check generated artifacts and run the authored activation, near-miss and causal-boundary fixtures in `tests/test_extension.py`. These tests check software behavior and known applicability boundaries; no live model quality or skill-learning claim follows from a pass.
