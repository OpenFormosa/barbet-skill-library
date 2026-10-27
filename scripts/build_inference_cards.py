#!/usr/bin/env python3
"""Attach source versions/hashes to original compact cards; check drift in CI."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    spec = json.loads((ROOT / 'testtime/card_sources.json').read_text())
    for card in spec['cards']:
        path = ROOT / 'skills' / card['source_skill'] / 'skill.json'
        source = json.loads(path.read_text())
        card['source_version'] = source['version']
        card['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        card['text_sha256'] = hashlib.sha256(card['text'].encode()).hexdigest()
    spec['token_budget_notice'] = 'Measured by the serving model tokenizer at inference; no character-count token guarantee.'
    payload = json.dumps(spec, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    target = ROOT / 'testtime/cards.json'
    if args.check:
        assert target.read_text() == payload, 'inference cards are out of date'
    else:
        target.write_text(payload)
    print('INFERENCE_CARDS_VALID', len(spec['cards']))


if __name__ == '__main__':
    main()
