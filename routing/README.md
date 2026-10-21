# Optional causal procedure router

`extension.select({"messages": [...], "tools": [...]})` suggests a minimal procedure after a more specific application router finds no applicable skill. It returns a skill name and input-only trigger evidence, or `None`. The registry itself does not prove a tool can fulfill the request: `tool-capability-triage` explicitly compares capabilities and permits clarification or a supported alternative.

This router uses English and Traditional Chinese lexical cues. It is a reproducible baseline, not a semantic classifier, safety policy, execution authorization or proof of answer correctness. A candidate must still satisfy the skill's Use When / Do Not Use When and the application's higher-priority instructions. No universal fallback is applied to noise, social turns or unsupported input roles. Do not infer domain correctness from a coverage percentage.

Only causal `messages` and `tools` are accepted. Do not pass target answers, later tool observations, source names, gold labels, splits or evaluation scores. Different answers to an identical input must receive the same routing decision. Keep the original task, tool schemas and response format; run answer compatibility checks separately and never use those checks to choose a route.

## Specific property questions before a concept fallback

Version `causal-procedure-extension/2` distinguishes a concrete entity's property from a definition: fictional `何謂白港塔的高度？` selects `factual-question-resolution`, while `何謂海拔高度？` selects `concept-explanation`. Digits inside an entity name do not by themselves establish an arithmetic task. Selection never depends on knowing the answer or whether a proposed value is correct.

Applications that already have a broad `什麼是` / `說明` concept rule can call `select_factual_property(causal_input)` immediately **before that fallback**, after their reviewed routes for tools, source-bound reading, translation, rewriting, code, calculations and other specific tasks. Convert its `skill` field to the application's route format. The hook returns `None` for unsupported phrasing, quotations, document cues, live-state wording, tool registries and multiple tasks. It is deliberately narrow; applications still own those priorities and the rest of their routing. Merely calling `select` after an earlier greedy concept match will not fix that match.

The general `select` function also uses this hook before its own concept fallback. It does not replace a source-aware application router. Test both activation and nearby nonactivation on the application's frozen causal inputs before adopting the new version; do not select a route by comparing reference answers or model scores. Further factual-answer and calibration checks are described in [the factual guidance](../docs/factual-calibration.md).

For SFT derivatives, pin the library commit and content hashes, keep inserted skill text out of supervised labels, preserve source `loss=false`, re-tokenize the entire sample without truncation, and recalculate selection budgets. Each skill block applies only to its next assistant prediction point. Local data consumers must preserve provenance, root grouping, splits and exclusions. This repository contains no training samples or target answers and does not publish a dataset update.

Run `python scripts/build.py --check` to check generated artifacts and run the authored activation, near-miss and causal-boundary fixtures in `tests/test_extension.py`. These tests check software behavior and known applicability boundaries; no live model quality or skill-learning claim follows from a pass.
