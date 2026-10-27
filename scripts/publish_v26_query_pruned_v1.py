"""Publish the value-preserving alias-pruned v26 release, then verify anonymously."""
import argparse
import json
import shutil

from scripts import build_memory_sft_v2 as base
from scripts import build_v26_complete_release as release
from scripts.prune_v26_query_aliases_v1 import STATE, DEST


def main(phase):
    summary = json.loads((STATE / 'summary.json').read_text())
    if not summary['all_single_turn_query_alias_holds_absent']:
        raise ValueError('normalized single-turn question gate missing')
    release.COMPACT = STATE; release.DEST = DEST
    release.publish(phase)
    if phase != 'prepare-publication': return
    reports = DEST / 'reports/v26-selected'
    shutil.copyfile(STATE / 'policy.json', reports / 'normalized-query-alias-holds.json')
    shutil.copyfile(base.WORK / 'reports/memory-harness-v2/v26-query-alias-hold-tests-v1.log', reports / 'v26-query-alias-hold-tests-v1.log')
    policy = json.loads((STATE / 'policy.json').read_text())
    note = (f"\nBefore publication, {len(policy['held_source_queries']):,} normalized single-turn question identities in OpenOrca/Orca-Math were found to retain multiple literal conditioning views. "
        f"All {policy['held_selected_views']:,} selected views of those questions are held. Inspected examples differed in surrounding or internal whitespace, line breaks, or case. "
        "Those differences do not create independent new question episodes, and PPL values conditioned on different literal inputs are not compared as if the inputs matched. "
        "No first-source answer is substituted. Future recovery requires a separately validated common-input policy and fresh measured scores where the conditioning changes. "
        "The original locally verified payload remains preserved; it was not published. The alias filter preserves every retained text, token, label and budget proof exactly.\n")
    card = DEST / 'README.md'; card.write_text(card.read_text() + note)
    report = reports / 'report.html'; report.write_text(report.read_text().replace('</html>',
        f'<p>發布前再隔離 {len(policy["held_source_queries"]):,} 個正規化後相同的單回合題目，合計 {policy["held_selected_views"]:,} 個不同輸入視圖。不能把空白、換行或大小寫差異當作新題目，也不能拿不同完整輸入下的 PPL 直接相比。這些題目等待一致輸入政策與重新評分；保留資料的文字、labels 與預算證明均未改變。</p></html>'))
    plan = json.loads((STATE / 'publication-plan.json').read_text())
    plan['normalized_query_alias_holds_included'] = True
    plan['manifest'] = [{'path': str(p.relative_to(DEST)), 'bytes': p.stat().st_size, 'sha256': base.file_sha(p)}
        for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'PUBLICATION.json']
    base.save(STATE / 'publication-plan.json', plan)
    print(json.dumps({'files': len(plan['manifest']), 'accepted': plan['accepted'],
                      'normalized_question_gate_bound': True}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare-publication', 'publish', 'verify']); args = p.parse_args()
    main(args.phase)
