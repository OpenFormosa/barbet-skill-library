"""Extend frozen family aliases through the fully checked NI scope-v2 wave."""
import json
from pathlib import Path

from scripts import build_memory_sft_v2 as base
from scripts import freeze_v26_release_folds_v1 as freezer
from scripts.package_natural_instructions_scope_v2 import STATE as NI
from scripts.rank_natural_instructions_v26_v1 import STATE as NI_SCORE
from scripts.freeze_v26_release_folds_v1 import STATE as ANCESTOR
from scripts.inventory_opencode_v1 import STATE as OPENCODE

STATE = OPENCODE / 'frozen-prior-v1'


def main():
    STATE.mkdir(exist_ok=True)
    source = json.loads((NI / 'public-package-summary.json').read_text())
    readback = json.loads((NI / 'final-scope-readback-v2.json').read_text())
    if not source['all_previous_folds_preserved'] or not readback['all_public_record_scope_gates_reread']:
        raise ValueError('NI release scope or prior folds unverified')
    proxy = STATE / 'prior-package-interface'; proxy.mkdir(exist_ok=True)
    # The generic freezer's older boolean name maps to the verified same gate.
    # Original package metadata stays byte-identical and hash-bound here.
    base.save(proxy / 'public-package-summary.json', {**source,
        'all_published_query_folds_preserved': source['all_previous_folds_preserved'],
        'original_summary_sha256': base.file_sha(NI / 'public-package-summary.json')})
    if not (proxy / 'public-index.sqlite').exists():
        (proxy / 'public-index.sqlite').symlink_to(NI / 'public-index.sqlite')
    base.save(STATE / 'binding.json', {'ni_scope_readback_sha256': base.file_sha(NI / 'final-scope-readback-v2.json'),
        'ni_public_index_sha256': base.file_sha(NI / 'public-index.sqlite'),
        'ancestor_summary_sha256': base.file_sha(ANCESTOR / 'summary.json'),
        'freezer_code_sha256': base.file_sha(base.REPO / 'scripts/freeze_v26_release_folds_v1.py'),
        'driver_sha256': base.file_sha(Path(__file__)), 'all_prior_and_losing_answer_aliases_preserved': True})
    freezer.STATE = STATE; freezer.PRIOR = proxy; freezer.PRIOR_SCORE = NI_SCORE; freezer.ANCESTOR = ANCESTOR
    freezer.main()


if __name__ == '__main__': main()
