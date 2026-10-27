"""External state with provenance checks, never an answer-correctness oracle."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


@dataclass(frozen=True)
class Evidence:
    id: str
    document_id: str
    document_hash: str
    start: int
    end: int
    quote: str
    status: str = 'source_span_verified_not_entailment'


class State:
    def __init__(self, task: str):
        self.task = task
        self.evidence: dict[str, Evidence] = {}
        self.hypotheses: list[str] = []
        self.failures: list[dict] = []
        self.receipts: list[dict] = []

    def add_evidence(self, document, start: int, end: int, quote: str):
        if type(start) is not int or type(end) is not int:
            raise ValueError('span indices must be integers')
        if not (0 <= start < end <= len(document.text)) or document.text[start:end] != quote:
            raise ValueError('quote does not match the original source span')
        eid = digest(f'{document.id}:{document.sha256}:{start}:{end}')[:20]
        e = Evidence(eid, document.id, document.sha256, start, end, quote)
        self.evidence[eid] = e
        return e

    def validate_evidence(self, eid, document):
        e = self.evidence[eid]
        return (e.document_id == document.id and e.document_hash == document.sha256
                and document.text[e.start:e.end] == e.quote)

    def record_execution(self, receipt: dict):
        # Called by the host executor, not exposed as a model tool. Receipt
        # semantics deliberately say public checks, never overall task success.
        required = {'executor', 'artifact_sha256', 'checks_sha256', 'passed', 'status'}
        if not required <= receipt.keys() or receipt['executor'] != 'isolated_python_v1':
            raise ValueError('invalid executor receipt')
        self.receipts.append(dict(receipt))

    def latest_public_pass(self, artifact: str, checks: str):
        return any(r['passed'] and r['artifact_sha256'] == digest(artifact)
                   and r['checks_sha256'] == digest(checks) for r in self.receipts)

    def handoff(self):
        return json.dumps({'task': self.task,
            'evidence': [asdict(e) for e in self.evidence.values()],
            'hypotheses_not_verified': self.hypotheses,
            'failed_attempts': self.failures,
            'execution_receipts_not_task_correctness': self.receipts},
            ensure_ascii=False, sort_keys=True)
