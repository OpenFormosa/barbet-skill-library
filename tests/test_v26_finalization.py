import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from scripts import finalize_v26_answer_selection as finalizer


class FinalizationTests(unittest.TestCase):
    def test_missing_premise_hold_keeps_separate_document_grounded_condition(self):
        rows=[{'source':'Open-Orca/OpenOrca','query_hash':'same-question','record_id':'bare'},
              {'source':'hotpotqa/hotpot_qa','query_hash':'same-question','record_id':'with-document'},
              {'source':'Open-Orca/OpenOrca','query_hash':'different-question','record_id':'other'}]
        with patch.object(finalizer,'EXTRA_SOURCE_QUERY_HOLDS',{('Open-Orca/OpenOrca','same-question')}):
            self.assertEqual([r['record_id'] for r in finalizer.eligible_source_conditions(rows)],['with-document','other'])

    def test_different_literal_tool_ids_are_not_identical_condition_bytes(self):
        a={'messages':[{'role':'user','content':'same question'},{'role':'tool','content':'same result','tool_call_id':'source_a','name':'lookup'}],'tools':[]}
        b=json.loads(json.dumps(a));b['messages'][1]['tool_call_id']='source_b'
        self.assertNotEqual(finalizer.strict_condition(a),finalizer.strict_condition(b))

    def test_stored_wrong_winner_cannot_be_finalized(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);score=root/'score';state=root/'state';state.mkdir()
            full=score/'scores/canonical-v2';full.mkdir(parents=True)
            candidate=sqlite3.connect(score/'candidates.sqlite');candidate.execute('CREATE TABLE groups(input_hash TEXT,candidates INTEGER)');candidate.execute("INSERT INTO groups VALUES ('g',2)");candidate.commit();candidate.close()
            for bucket in range(8):
                (full/f'bucket-{bucket}.summary.json').write_text(json.dumps({'status':'BUCKET_COMPLETE','limited_pilot':False,'protocol_sha256':'p'}))
                shard=sqlite3.connect(full/f'bucket-{bucket}.sqlite')
                shard.execute('CREATE TABLE decisions(input_hash TEXT,status TEXT,winner TEXT,report_json TEXT)');shard.execute('CREATE TABLE scores(candidate_identity TEXT,input_hash TEXT,record_id TEXT,score_json TEXT)')
                if bucket==0:
                    shard.execute("INSERT INTO decisions VALUES ('g','selected','bad','{}')")
                    for rid,nll in [('bad',3.0),('best',1.0)]:
                        s={'input_hash':'g','candidate_identity':rid,'record_id':rid,'eligible':True,'protocol_sha256':'p','status':'scored','mean_nll':nll,'answer_tokens':5}
                        shard.execute('INSERT INTO scores VALUES (?,?,?,?)',(rid,'g',rid,json.dumps(s)))
                shard.commit();shard.close()
            db=sqlite3.connect(state/'selection.sqlite');db.row_factory=sqlite3.Row;db.execute('ATTACH DATABASE ? AS candidates',(str(score/'candidates.sqlite'),))
            with patch.object(finalizer,'SCORE',score),patch.object(finalizer,'STATE',state):
                with self.assertRaisesRegex(ValueError,'winner is not minimum'):finalizer.merge_decisions(db)
            db.close()


if __name__=='__main__':unittest.main()
