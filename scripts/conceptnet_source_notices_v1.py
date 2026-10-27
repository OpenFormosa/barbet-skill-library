"""Preserve required source credits and the original WordNet license notice.

Only publication notices are added. Reviewed messages, targets, per-assertion
license metadata and compiled arrays remain byte-for-byte unchanged.
"""
from html.parser import HTMLParser
import json
from pathlib import Path

from scripts import build_memory_sft_v2 as base
from scripts.fetch_conceptnet57_primary_v1 import STATE as SOURCE,RAW


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__();self.parts=[]

    def handle_data(self,data):self.parts.append(data)

    def handle_endtag(self,tag):
        if tag=='p':self.parts.append('\n\n')


def write_notices(dest,report_dir):
    plan=json.loads((SOURCE/'source-plan.json').read_text())
    page=next(p for p in plan['primary_references'] if p['file']=='copying-and-sharing.html')
    original=RAW/page['file']
    if base.file_sha(original)!=page['sha256']:raise ValueError('original ConceptNet attribution page changed')
    html=original.read_text();start=html.index('<p>WordNet Release 3.0</p>');end=html.index('<div class="markdown-heading">',start)
    parser=Paragraphs();parser.feed(html[start:end]);text=''.join(parser.parts).strip()+'\n'
    required=['WordNet Release 3.0','Copyright 2006 by Princeton University','ALL copies',
        'NO REPRESENTATIONS OR WARRANTIES','LICENSEE agrees to preserve']
    if any(s not in text for s in required):raise ValueError('complete original WordNet notice not extracted')
    folder=dest/'licenses';folder.mkdir(exist_ok=True)
    notice=folder/'ConceptNet-WordNet-original-notice.txt'
    if notice.exists() and notice.read_text()!=text:raise ValueError('existing source notice differs')
    notice.write_text(text)
    base.save(report_dir/'source-notices-binding.json',{
        'driver_sha256':base.file_sha(Path(__file__)),'pinned_original_page_url':page['url'],
        'pinned_original_page_sha256':page['sha256'],'wordnet_notice_sha256':base.file_sha(notice),
        'version_heading_preserved_exactly':True,'wordnet_source_assertion_dataset':'/d/wordnet/3.1',
        'source_page_reproduces_notice_titled':'WordNet Release 3.0',
        'current_primary_license_page':'https://wordnet.princeton.edu/license-and-commercial-use',
        'native_messages_targets_tokens_labels_changed':False,'model_requests':0})
    return '''

### ConceptNet and underlying contributor credits

This release uses ConceptNet 5 data compiled by the Commonsense Computing Initiative, including contributions from Wikimedia's Wiktionary communities and Princeton University's WordNet. Research attribution: Robyn Speer, Joshua Chin, and Catherine Havasi (2017), “ConceptNet 5.5: An Open Multilingual Graph of General Knowledge,” AAAI 31. ConceptNet's complete source compilation is published under CC-BY-SA-4.0; exact contributing sources and explicit licenses remain recorded for every included assertion. Source and attribution: https://conceptnet.io/ and https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet .

The full Princeton copyright notice, permission conditions and warranty disclaimer reproduced by ConceptNet are included in [licenses/ConceptNet-WordNet-original-notice.txt](licenses/ConceptNet-WordNet-original-notice.txt). Its original “WordNet Release 3.0” heading is preserved; the source assertion dataset is still recorded as /d/wordnet/3.1. The primary WordNet license page is https://wordnet.princeton.edu/license-and-commercial-use . This publication supplement preserves source notices and does not change the source facts, reviewed targets, per-assertion license labels or token arrays.
'''
