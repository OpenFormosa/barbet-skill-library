import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from scripts import build_memory_sft_wave2 as wave


class FrozenSplitTests(unittest.TestCase):
    def test_new_bridge_between_frozen_folds_is_held(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'normalized').mkdir()
            db=sqlite3.connect(p/'frozen-aliases.sqlite')
            db.execute('CREATE TABLE aliases(node TEXT PRIMARY KEY,family_id TEXT,split TEXT)')
            db.execute('CREATE TABLE conflicts(node TEXT PRIMARY KEY)')
            db.executemany('INSERT INTO aliases VALUES (?,?,?)',[('family:a','old-a','train'),('family:b','old-b','validation')]);db.commit();db.close()
            rows=[{'family_id':x,'query_hash':'same-query','template_family_id':'shared'} for x in ['a','b']]
            (p/'normalized/00000.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (p/'normalize-jobs.json').write_text('[{}]');(p/'normalize-summary.json').write_text('{}')
            with patch.object(wave,'STATE',p),contextlib.redirect_stdout(io.StringIO()):wave.dedup()
            summary=json.loads((p/'dedup-summary.json').read_text())
            self.assertEqual(summary['held_cross_wave_split_bridge'],2)
            self.assertEqual(summary.get('selected',0),0)
            check=sqlite3.connect(p/'frozen-aliases.sqlite')
            self.assertEqual(check.execute('SELECT split FROM aliases ORDER BY node').fetchall(),[('train',),('validation',)])
            check.close()

    def test_incomplete_wave_cannot_assign_splits(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(wave,'STATE',Path(folder)):
            with self.assertRaisesRegex(ValueError,'incomplete'):wave.dedup()


if __name__=='__main__':unittest.main()
