"""One actual minimum-v26-PPL OpenCode winner, with all prior folds preserved."""
import argparse
import json

from scripts import build_memory_sft_v2 as base
from scripts import finalize_v26_answer_selection as selector
from scripts.inventory_opencode_v1 import STATE as SOURCE
from scripts.rank_opencode_v26_v1 import STATE as SCORE
from scripts.freeze_opencode_prior_v1 import STATE as FROZEN

STATE = SOURCE / 'selected-v1'
OUT = base.WORK / 'exports/memory-harness-v2-opencode-selected-v1'


def finalize():
    frozen = json.loads((FROZEN / 'summary.json').read_text())
    if frozen['frozen_database_sha256'] != base.file_sha(FROZEN / 'frozen-aliases.sqlite'):
        raise ValueError('prior family aliases changed')
    if not frozen['all_candidate_aliases_included'] or not frozen['all_ancestral_folds_preserved']:
        raise ValueError('prior alias coverage incomplete')
    selector.STATE = STATE; selector.OUT = OUT; selector.SCORE = SCORE; selector.SECOND = FROZEN
    selector.ROWS_PER_SHARD = 20000; selector.main()
    base.save(STATE / 'release-scope.json', {'scope': 'New OpenCode wave only; requires new winner diagnosis, complete export and cumulative publication.',
        'frozen_prior_sha256': base.file_sha(FROZEN / 'summary.json'),
        'all_minima_independently_recomputed': True, 'all_first_negative_source_queries_held': True,
        'unequal_literal_input_aliases_held_before_scoring': True, 'source_test_code_executed': False,
        'source_test_results_not_independently_confirmed': True, 'not_yet_published': True, 'weights_trained': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['finalize', 'compile']); args = p.parse_args()
    if args.phase == 'finalize': finalize()
    else:
        if not (STATE / 'release-scope.json').exists(): raise ValueError('selection incomplete')
        base.STATE = STATE; base.OUT = OUT; base.compile_all(16, False)
