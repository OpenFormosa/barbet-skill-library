"""Deterministic lexical retrieval for English identifiers and Chinese text.

No embedding model, fitting labels, or claim that BM25 scores are probabilities.
Original passages are returned intact; ranking normalization never changes them.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
import re
import unicodedata


def terms(text: str) -> list[str]:
    text = unicodedata.normalize('NFKC', text).casefold()
    result = re.findall(r'[a-z_][a-z_0-9]*|\d+(?:\.\d+)?', text)
    for run in re.findall(r'[\u3400-\u9fff]+', text):
        result.extend(run)
        result.extend(run[i:i + 2] for i in range(len(run) - 1))
    return result


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str
    title: str = ''

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


class LexicalIndex:
    def __init__(self, documents: list[Document]):
        if len({d.id for d in documents}) != len(documents):
            raise ValueError('duplicate document id')
        self.documents = documents
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for i, d in enumerate(documents):
            counts = Counter(terms(d.title + '\n' + d.text))
            self.lengths.append(sum(counts.values()))
            for word, n in counts.items():
                self.postings[word].append((i, n))
        self.average = sum(self.lengths) / max(1, len(documents)) or 1

    def search(self, query: str, k: int = 4, exclude: set[str] | None = None):
        scores: dict[int, float] = defaultdict(float)
        n = len(self.documents)
        for word in set(terms(query)):
            posting = self.postings.get(word, ())
            idf = math.log(1 + (n - len(posting) + .5) / (len(posting) + .5))
            # Single CJK characters provide recall, but exact two-character
            # entities/terms should outrank accidental common-character overlap.
            # This weight is an explicit lexical heuristic, not learned.
            weight = .25 if len(word) == 1 and '\u3400' <= word <= '\u9fff' else 1.0
            if len(word) == 2 and any(c in '的有在了與和及於幾' for c in word):
                weight *= .25
            for i, tf in posting:
                norm = 1.2 * (1 - .75 + .75 * self.lengths[i] / self.average)
                scores[i] += weight * idf * tf * 2.2 / (tf + norm)
        ordered = sorted(scores, key=lambda i: (-scores[i], self.documents[i].id))
        return [(self.documents[i], scores[i]) for i in ordered
                if self.documents[i].id not in (exclude or set())][:max(0, k)]


def pack_documents(ranked, token_count, budget: int):
    """Keep whole evidence records and count the actual target tokenizer.

    Returns selected documents and IDs that did not fit. A large paragraph is
    never silently chopped, which could erase its negation or antecedent.
    """
    selected, omitted, blocks = [], [], []
    for doc, score in ranked:
        block = f'<source id="{doc.id}" sha256="{doc.sha256}">\n{doc.title}\n{doc.text}\n</source>'
        if token_count('\n\n'.join(blocks + [block])) > budget:
            omitted.append(doc.id)
            continue
        selected.append({'id': doc.id, 'score': score, 'sha256': doc.sha256,
                         'source': doc.source, 'text': doc.text, 'title': doc.title})
        blocks.append(block)
    return '\n\n'.join(blocks), selected, omitted
