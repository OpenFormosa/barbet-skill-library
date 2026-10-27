"""Versioned release driver including recovered raw answer alternatives."""
import argparse
import fcntl
import json
import shutil

from scripts import build_memory_sft_v2 as base
from scripts.extend_v26_overlap_selection import STATE as SCORE, RECOVERY

STATE = base.WORK / 'reports/memory-harness-v2-v26-complete-selected-v1'
OUT = base.WORK / 'exports/memory-harness-v2-v26-complete-selected-v1'
PUBLIC = base.WORK / 'exports/memory-harness-v2-v26-complete-selected-v1-public'
DEST = base.WORK / 'exports/memory-harness-v2-v26-complete-public-v1'
COMPACT = STATE / 'compact-v1'


def finalize():
    from scripts import finalize_v26_answer_selection as selector
    selector.SCORE = SCORE; selector.STATE = STATE; selector.OUT = OUT
    holds = SCORE / 'source-premise-holds-v1.json'
    selector.EXTRA_SOURCE_QUERY_HOLDS = {(r['source'], r['query_hash']) for r in json.loads(holds.read_text())['holds']}
    selector.EXTRA_SOURCE_QUERY_HOLD_RECEIPT = holds
    selector.main()


def build(phase, workers):
    from scripts import build_v26_selected_release as release
    release.STATE = STATE; release.OUT = OUT; release.PUBLIC = PUBLIC
    if phase == 'audit': release.audit()
    elif phase == 'compile': release.compile_all(workers)
    else: release.package_all(workers)


def compact():
    from scripts import compact_v26_selected_release as packing
    packing.SELECTED = STATE; packing.SOURCE = PUBLIC
    packing.STATE = COMPACT; packing.DEST = DEST
    packing.main()


def publish(phase):
    from scripts import publish_v26_selected_release as publisher
    publisher.STATE = COMPACT; publisher.SELECTED = STATE
    publisher.DEST = DEST; publisher.SCORE = SCORE; publisher.configure()
    with (publisher.FIRST / 'publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == 'publish': publisher.transport.publish(); return
        if phase == 'verify': publisher.transport.verify(); return
        publisher.prepare()
        # These are metadata-only additions before the publication plan is
        # frozen remotely. All data values and measured PPL remain unchanged.
        reports = DEST / 'reports/v26-selected'
        recovery = json.loads((RECOVERY / 'summary.json').read_text())
        base.save(reports / 'raw-query-overlap-recovery.json', {
            k: v for k, v in recovery.items() if k != 'files'})
        for src, name in [
            (SCORE / 'candidate-policy-binding.json', 'extended-candidate-binding.json'),
            (SCORE / 'scores/canonical-v2/reuse-summary.json', 'score-reuse-summary.json'),
            (SCORE / 'source-premise-holds-v1.json', 'source-premise-holds.json'),
            (publisher.FIRST / 'v26-overlap-score-reuse-tests.log', 'v26-overlap-score-reuse-tests.log')]:
            shutil.copyfile(src, reports / name)
        shutil.copyfile(publisher.FIRST / 'v26-finalization-tests-v3.log', reports / 'v26-finalization-tests-v3.log')
        counts = recovery['counts']
        note = (f"\nThe candidate pool also rechecks {counts['seen_query_overlap']:,} raw rows previously skipped only because their first query had already appeared. "
                f"It stages {counts['new_eligible_answer_variant']:,} new eligible answer candidates before global identity deduplication; these are not extra accepted episodes. "
                "Original quality, legacy, evaluation and first-negative holds remain in force. "
                "Existing numerical scores are reused only after all original source-file hashes, candidate pointers and model/scoring protocol hashes match. "
                "The extended candidate binding and score-reuse receipt retain the original score-database hashes. New alternatives and formerly unscored singleton competitors receive actual v26 inference.\n")
        card = DEST / 'README.md'; card.write_text(card.read_text() + note)
        plan = json.loads((COMPACT / 'publication-plan.json').read_text())
        plan['manifest'] = [
            {'path': str(p.relative_to(DEST)), 'bytes': p.stat().st_size, 'sha256': base.file_sha(p)}
            for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'PUBLICATION.json']
        plan['raw_query_overlap_recovery_included'] = True
        base.save(COMPACT / 'publication-plan.json', plan)
        print(json.dumps({'metadata_complete': True, 'files': len(plan['manifest']),
                          'bytes': sum(f['bytes'] for f in plan['manifest'])}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['finalize', 'audit', 'compile', 'package', 'compact',
                                   'prepare-publication', 'publish', 'verify'])
    p.add_argument('--workers', type=int, default=24); args = p.parse_args()
    if not 1 <= args.workers <= 32: raise ValueError('CPU worker bound')
    if args.phase == 'finalize': finalize()
    elif args.phase in ['audit', 'compile', 'package']: build(args.phase, args.workers)
    elif args.phase == 'compact': compact()
    else: publish(args.phase)
