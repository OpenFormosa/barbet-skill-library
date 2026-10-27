"""Pinned Wikipedia paragraphs for SYNTH, retrieved using only the question.

Source-provided article URLs define a restricted candidate pool. This is not a
global Wikipedia search, and it does not prove that an upstream answer is true.
"""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

import requests

from barbet_inference.memory import MemoryItem, canonical, compose_memory_prompt
from barbet_inference.sft import Rejected, RESERVED, augment_system, normalize_messages
from scripts import build_memory_sft_v2 as base
from testtime.retrieval import Document, LexicalIndex

STATE = base.WORK / 'reports/memory-harness-v2-synth-wikipedia-v1'
RAW = base.RAW / 'scale10m-synth-wikipedia-pilot'
USER_AGENT = 'BarbetMemoryDatasetResearch/0.1 (https://github.com/OpenFormosa/barbet-skill-library)'
API = 'https://en.wikipedia.org/w/api.php'
LICENSE = 'https://creativecommons.org/licenses/by-sa/4.0/'


def wikipedia_title(url):
    if not isinstance(url, str): raise Rejected('missing_wikipedia_seed_url')
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.netloc != 'en.wikipedia.org' or not parsed.path.startswith('/wiki/'):
        raise Rejected('outside_english_wikipedia_pilot')
    title = unquote(parsed.path[len('/wiki/'):]).replace('_', ' ').strip()
    if not title or ':' in title or parsed.query:
        raise Rejected('nonarticle_or_ambiguous_seed_url')
    return title


class ArticleText(HTMLParser):
    """Render text blocks, retaining math alt text and table/subscript content.

    Script/style bytes and edit links are excluded. Images are not downloaded or
    redistributed. Paragraph selection, rather than this renderer, limits input.
    """
    BLOCKS = {'p', 'div', 'section', 'li', 'ul', 'ol', 'table', 'tr',
              'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'pre'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []; self.suppressed = []; self.math_count = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if self.suppressed:
            if tag not in {'img', 'br', 'hr', 'input', 'meta', 'link', 'wbr'}:
                self.suppressed.append(tag)
            return
        if tag in {'script', 'style'} or 'mw-editsection' in attrs.get('class', '').split():
            self.suppressed.append(tag); return
        if tag == 'math':
            alt = attrs.get('alttext')
            if not alt: raise Rejected('math_without_preserved_alt_text')
            self.parts.append(' ' + alt + ' '); self.math_count += 1
            self.suppressed.append(tag); return
        if tag in self.BLOCKS: self.parts.append('\n\n')
        if tag in {'td', 'th'}: self.parts.append(' | ')
        if tag == 'br': self.parts.append('\n')
        if tag == 'sub': self.parts.append('_(')
        if tag == 'sup': self.parts.append('^(')
        if tag == 'img' and attrs.get('alt'): self.parts.append(' [image alt: ' + attrs['alt'] + '] ')

    def handle_endtag(self, tag):
        if self.suppressed:
            if tag in self.suppressed:
                at = len(self.suppressed) - 1 - self.suppressed[::-1].index(tag)
                del self.suppressed[at:]
            return
        if tag in self.BLOCKS: self.parts.append('\n\n')
        if tag in {'sub', 'sup'}: self.parts.append(')')

    def handle_data(self, data):
        if not self.suppressed: self.parts.append(data)

    def paragraphs(self):
        text = ''.join(self.parts)
        return [re.sub(r'\s+', ' ', part).strip() for part in re.split(r'\n\s*\n', text)
                if part.strip()]


def render_article(html):
    parser = ArticleText(); parser.feed(html); parser.close()
    if parser.suppressed: raise Rejected('unterminated_article_markup')
    paragraphs = parser.paragraphs()
    if not paragraphs: raise Rejected('empty_article_text')
    return paragraphs, parser.math_count


def fetch_article(title):
    folder = RAW / base.sha(title); folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / 'receipt.json'
    if receipt.exists():
        item = json.loads(receipt.read_text())
        for key in ['lookup', 'parse']:
            f = folder / (key + '.json')
            if base.file_sha(f) != item[key + '_sha256']: raise ValueError('pinned article bytes changed')
        return item
    # An existing incomplete fetch is transport evidence, not a model review.
    # Keep a successful lookup to pin the same revision on a later retry.
    lookup = folder / 'lookup.json'; parsed_path = folder / 'parse.json'
    with requests.Session() as session:
        session.headers.update({'User-Agent': USER_AGENT})
        if not lookup.exists():
            result = session.get(API, params={'action': 'query', 'format': 'json', 'prop': 'info',
                'titles': title, 'redirects': 1, 'maxlag': 5}, timeout=(15, 60))
            result.raise_for_status(); data = result.json()
            if 'error' in data: raise Rejected('wikipedia_api_error')
            lookup.write_bytes(result.content)
        pages = list(json.loads(lookup.read_text())['query']['pages'].values())
        if len(pages) != 1 or 'missing' in pages[0] or pages[0].get('ns') != 0:
            raise Rejected('missing_article_or_namespace_redirect')
        page = pages[0]; revision = page['lastrevid']
        if not parsed_path.exists():
            result = session.get(API, params={'action': 'parse', 'format': 'json', 'oldid': revision,
                'prop': 'text|revid|displaytitle', 'maxlag': 5}, timeout=(15, 60))
            result.raise_for_status(); data = result.json()
            if 'error' in data: raise Rejected('wikipedia_pinned_parse_error')
            parsed_path.write_bytes(result.content)
    parsed = json.loads(parsed_path.read_text())['parse']
    if parsed['revid'] != revision: raise ValueError('article revision mismatch')
    paragraphs, math_count = render_article(parsed['text']['*'])
    item = {'requested_title': title, 'title': page['title'], 'page_id': page['pageid'],
            'revision': revision, 'source': 'https://en.wikipedia.org/w/index.php?oldid=' + str(revision),
            'license': 'cc-by-sa-4.0', 'license_url': LICENSE,
            'attribution': 'Wikipedia contributors; linked article history identifies authors.',
            'lookup_sha256': base.file_sha(lookup), 'parse_sha256': base.file_sha(parsed_path),
            'paragraphs': len(paragraphs), 'math_alt_texts_preserved': math_count,
            'renderer_sha256': base.file_sha(Path(__file__)),
            'transformation': 'Rendered HTML to text blocks with whitespace normalization, math alt text and sub/sup markup; no images redistributed.',
            'additional_import_attribution_reviewed': False}
    base.save(receipt, item); return item


def article_documents(item):
    if item['renderer_sha256'] != base.file_sha(Path(__file__)): raise ValueError('article renderer changed')
    path = RAW / base.sha(item['requested_title']) / 'parse.json'
    if base.file_sha(path) != item['parse_sha256']: raise ValueError('article parse changed')
    data = json.loads(path.read_text())['parse']; paragraphs, _ = render_article(data['text']['*'])
    return [Document(f"wiki-{item['page_id']}-{item['revision']}-{i}", p, item['source'], item['title'])
            for i, p in enumerate(paragraphs)]


def retrieve(query, articles):
    # No answer argument exists in this function. All selected blocks are whole.
    documents = [doc for article in articles for doc in article_documents(article)]
    ranked = LexicalIndex(documents).search(query, 16)
    if not ranked: raise Rejected('no_query_retrieval_match')
    if any(len(doc.text) > 24000 for doc, _ in ranked): raise Rejected('oversize_whole_retrieved_block')
    return ranked, len(documents)


def adapt(raw, source, index, articles, skills):
    query, answer = raw['query'], raw['synthetic_answer']
    if not query.strip() or not answer.strip() or RESERVED.search(query + answer):
        raise Rejected('missing_text_or_reserved_literal')
    ranked, indexed = retrieve(query, articles)
    items = [MemoryItem(d.id, d.title + '\n' + d.text, d.source,
                       observation_origin='query_only_retrieval_from_pinned_source_linked_articles') for d, _ in ranked]
    qh = base.sha(base.norm(query)); placement = 'front' if int(qh[:8], 16) % 2 else 'back'
    prompt = compose_memory_prompt(query, items, placement=placement)
    messages = augment_system(normalize_messages([{'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': answer}], [], {}), skills[0])
    rid = base.sha(canonical({'version': 'memory-harness-sft-v2', 'messages': messages, 'tools': []}))
    family = base.sha('query-family:' + base.norm(query, numbers=True))
    return {'record_id': rid, 'query_hash': qh, 'family_id': family, 'source_family_id': family,
        'template_family_id': family, 'source_task_key': qh, 'split': 'unassigned', 'domain': 'reading',
        'source': source['repo'], 'upstream_source': source['repo'], 'license': 'cc-by-sa-4.0',
        'messages': messages, 'tools': [], 'memory': {'documents': [i.__dict__ for i in items]},
        'skills': skills[1], 'skill_ids': base.ROUTES['reading'],
        'verification_tier': 'upstream_synthetic_answer_with_actual_retrieved_evidence_not_semantically_verified',
        'observation_origin': 'query_only_retrieval_from_pinned_source_linked_articles',
        'memory_placement': placement, 'actual_retrieval': True, 'recorded_error_followup': False,
        'assistant_turns': 1, 'tool_observations': 0, 'locally_executed_source_program': False,
        'source_refs': {'repo': source['repo'], 'revision': source['revision'], 'file': source['path'],
            'file_sha256': source['sha256'], 'row_index': index, 'upstream_id': raw['synth_id'],
            'source_projection_sha256': base.sha(canonical(raw)),
            'source_projection_excludes_reasoning_and_script': True,
            'attribution_url': f"https://huggingface.co/datasets/{source['repo']}/tree/{source['revision']}",
            'synthetic_output_license': 'cc-by-4.0', 'memory_license': 'cc-by-sa-4.0',
            'license_url': LICENSE, 'original_seed_url': raw['query_seed_url'],
            'original_seed_text_sha256': base.sha(raw['query_seed_text']),
            'additional_seed_url': raw['additional_seed_url'], 'articles': articles,
            'retrieval': {'query': query, 'query_sha256': base.sha(query), 'indexed_blocks': indexed,
                'pool': 'only the source-provided one or two Wikipedia articles',
                'top_k': 16, 'ranked': [{'id': d.id, 'sha256': d.sha256, 'score': score} for d, score in ranked],
                'answer_used_as_query': False, 'whole_blocks_retained': True,
                'index_driver_sha256': base.file_sha(base.REPO / 'testtime/retrieval.py')},
            'upstream_seed_text_used_as_evidence': False, 'new_semantic_generation': False,
            'source_question_answer_unchanged': True,
            'transformations': ['pin source-linked articles', 'question-only paragraph retrieval',
                'one causal memory placement', 'skill prefill', 'unchanged public answer', 'assistant-only loss']}}
