"""Load authored skill definitions from the checkout or installed wheel."""
from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path

from testtime.contracts import strict_json


class SkillLibrary:
    def __init__(self, path=None):
        if path is not None:
            root = Path(path)
            self.root = root / 'skills' if (root / 'skills').is_dir() else root
        else:
            installed = files('barbet_inference').joinpath('skill_data')
            self.root = installed if installed.is_dir() else Path(__file__).resolve().parents[1] / 'skills'
        if not self.root.is_dir():
            raise ValueError('skill directory not found')
        self._skills = {}
        for folder in sorted(self.root.iterdir(), key=lambda p: p.name):
            source = folder.joinpath('skill.json')
            if not folder.is_dir() or not source.is_file():
                continue
            raw = source.read_bytes()
            value = strict_json(raw.decode('utf-8'))
            if value.get('name') != folder.name or not isinstance(value.get('procedure'), str):
                raise ValueError('invalid skill definition: ' + folder.name)
            self._skills[folder.name] = {**value, 'source_sha256': hashlib.sha256(raw).hexdigest()}
        if not self._skills:
            raise ValueError('skill directory is empty')

    def list(self):
        return [{k: value[k] for k in ('name', 'version', 'scope', 'description', 'source_sha256')}
                for value in self._skills.values()]

    def get(self, name):
        if name not in self._skills:
            raise ValueError('unknown skill: ' + str(name))
        return dict(self._skills[name])

    def search(self, query, *, top_k=3):
        """Input-only lexical retrieval; scores are not relevance probabilities."""
        from testtime.retrieval import Document, LexicalIndex
        if not isinstance(query,str) or not query.strip() or type(top_k) is not int or top_k<1:
            raise ValueError('nonempty query and positive top_k required')
        aliases={'coding':'code python function program implementation tests debugging',
                 'agent':'tools function calling arguments action observation workflow',
                 'math':'math arithmetic numbers quantities rates units calculation',
                 'reading':'documents reading evidence retrieval sources passages memory',
                 'science':'science experiment hypothesis evidence explanation',
                 'writing':'writing text translation summarization revision'}
        docs=[Document(name, ' '.join(str(value[k]) for k in ('description','use_when','procedure'))+
                       ' '+aliases.get(value.get('scope'),''),value['source_sha256'],name.replace('-',' '))
              for name,value in self._skills.items()]
        return [{**self.get(d.id),'retrieval_score':score,'selection_basis':'query_only_lexical'}
                for d,score in LexicalIndex(docs).search(query,top_k)]

    def render(self, names):
        parts, provenance = [], []
        for name in dict.fromkeys(names):
            value = self.get(name)
            parts.append('\n'.join([f'Skill: {name} (version {value["version"]})'] +
                [f'{key}: {value[key]}' for key in
                 ('use_when', 'do_not_use_when', 'inputs', 'procedure', 'output_contract', 'failure', 'validation')]))
            provenance.append({k: value[k] for k in ('name', 'version', 'source_sha256')})
        return '\n\n'.join(parts), provenance
