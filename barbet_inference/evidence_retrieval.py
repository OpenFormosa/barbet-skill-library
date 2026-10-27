"""Question-only lexical retrieval that keeps whole nearby evidence blocks.

The caller supplies a permitted, provenance-bound document pool and tokenizer.
This module never generates facts, executes source text or reads target answers.
"""
from collections import OrderedDict
import hashlib

from .memory import MemoryItem, canonical, memory_block
from testtime.retrieval import Document, LexicalIndex, terms

STOP_WORDS = set('a an the is are was were be been being am to of in on at by for and or if so as with from that this these those it its they their them he she his her we our you your i me my not no yes can could would should will may might have has had do does did than then there here which who what when where how why each all any some also about more most such very only into over under through between after before during without within both other same different please tell explain actually like hey'.split())


def _search_text(value):
    return ' '.join(w for w in terms(value) if w not in STOP_WORDS)


def _chunks(documents, target_chars):
    by_source = OrderedDict()
    for document in documents:
        by_source.setdefault(document.source, []).append(document)
    chunks = []; members = {}; positions = {}
    for source, paragraphs in by_source.items():
        current = []
        def emit():
            text = '\n\n'.join(d.text for d in current)
            identifier = 'evidence-' + hashlib.sha256(canonical([source, [d.id for d in current], text]).encode()).hexdigest()
            members[identifier] = [d.id for d in current]
            positions[identifier] = len(chunks)
            chunks.append(Document(identifier, text, source, current[0].title))
        for paragraph in paragraphs:
            if current and (sum(len(d.text) for d in current) + len(paragraph.text) + 2 * len(current) > target_chars):
                emit(); current = []
            current.append(paragraph)
        if current: emit()
    return chunks, members, positions


def retrieve_with_neighbors(query, documents, token_count, *, max_tokens=12000,
                            top_k=8, neighbor_radius=1, target_chunk_chars=2400,
                            include_all_if_fits=True,
                            observation_origin='local_deterministic_retrieval'):
    """Return original complete blocks plus exact memory-envelope budget evidence.

    Nearby paragraphs may contain dates, antecedents or negations absent from a
    matching paragraph. Neighbors never cross an article/source boundary. The
    selected memory is in original source order, and oversize blocks are skipped
    explicitly, never truncated. Prompt and answer budgets remain caller checks.
    """
    if not isinstance(query, str) or not query.strip(): raise ValueError('empty retrieval question')
    if any(type(v) is not int or v < 1 for v in [max_tokens, top_k, target_chunk_chars]):
        raise ValueError('invalid evidence retrieval bounds')
    if type(neighbor_radius) is not int or not 0 <= neighbor_radius <= 2 or top_k > 64:
        raise ValueError('unbounded evidence neighborhood')
    if len({d.id for d in documents}) != len(documents): raise ValueError('duplicate paragraph identity')
    chunks, members, positions = _chunks(documents, target_chunk_chars)
    def envelope(values):
        return memory_block([MemoryItem(d.id, d.title + '\n' + d.text, d.source,
            observation_origin=observation_origin) for d in values], turn=0)
    def size(values): return token_count(envelope(values))
    if size([]) > max_tokens: raise ValueError('budget cannot hold the empty memory envelope')
    query_terms = _search_text(query)
    index = LexicalIndex([Document(d.id, _search_text(d.text), d.source, _search_text(d.title)) for d in chunks])
    ranked = index.search(query_terms, top_k)
    method = 'all_evidence_fits' if include_all_if_fits and size(chunks) <= max_tokens else 'question_lexical_with_neighbors'
    selected = []; omitted = []; considered = set()
    if method == 'all_evidence_fits': selected = chunks
    else:
        for hit, _ in ranked:
            center = positions[hit.id]
            offsets = [0] + [v for distance in range(1, neighbor_radius + 1) for v in [-distance, distance]]
            for delta in offsets:
                i = center + delta
                if not 0 <= i < len(chunks): continue
                item = chunks[i]
                if item.source != hit.source or item.id in considered: continue
                considered.add(item.id)
                proposal = sorted(selected + [item], key=lambda d: positions[d.id])
                if size(proposal) <= max_tokens: selected = proposal
                else: omitted.append(item.id)
    serialized = envelope(selected)
    trace = {'method': method, 'query_sha256': hashlib.sha256(query.encode()).hexdigest(),
        'answer_used_as_query': False, 'source_paragraphs': len(documents), 'indexed_chunks': len(chunks),
        'top_k': top_k, 'neighbor_radius': neighbor_radius, 'target_chunk_chars': target_chunk_chars,
        'memory_tokens': token_count(serialized), 'memory_token_budget': max_tokens,
        'memory_envelope_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
        'ranked_hits': [{'id': d.id, 'score': score} for d, score in ranked],
        'selected': [{'id': d.id, 'source': d.source, 'sha256': d.sha256,
                      'source_paragraph_ids': members[d.id]} for d in selected],
        'whole_chunks_omitted_for_budget': omitted, 'source_order_preserved': True,
        'source_text_truncated': False}
    return selected, trace
