import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
from scripts import compact_v26_selected_release as compact


class V26CompactionTests(unittest.TestCase):
    def inputs(self,source,reverse=False):
        tables={
            'records':pa.Table.from_pylist([{'record_id':'a','source_refs_json':json.dumps({'answer_selection':{'method':'minimum_v26_conditional_answer_ppl'}})},
                                          {'record_id':'b','source_refs_json':json.dumps({'answer_selection':{'method':'only_eligible_answer'}})}]),
            'tokenized':pa.Table.from_pylist([{'record_id':'a','input_ids':[1,4,5],'labels':[-100,4,5]},
                                            {'record_id':'b','input_ids':[1,7,8],'labels':[-100,-100,8]}]),
            'minicpm5_tokenized':pa.Table.from_pylist([{'record_id':'a','input_ids':[9,10,11],'labels':[-100,10,11]},
                                                    {'record_id':'b','input_ids':[9,12,13],'labels':[-100,-100,13]}])}
        if reverse:tables['tokenized']=tables['tokenized'].take(pa.array([1,0]))
        for name,table in tables.items():
            p=source/'data'/compact.WAVE/name/'train-00000.parquet';p.parent.mkdir(parents=True);pq.write_table(table,p)
        return tables

    def test_scored_and_singleton_rows_preserve_all_arrays_across_new_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=root/'state';state.mkdir();source=root/'source';dest=root/'dest';tables=self.inputs(source)
            with patch.multiple(compact,STATE=state,SOURCE=source,DEST=dest,ROWS=1),patch.multiple(compact.exact,STATE=state,DEST=dest):
                r=compact.job('train')
            self.assertEqual(r['counts']['accepted'],2)
            self.assertEqual(r['counts']['selection:minimum_v26_conditional_answer_ppl'],1)
            self.assertEqual(r['counts']['selection:only_eligible_answer'],1)
            for name,expected in tables.items():
                restored=pa.concat_tables([pq.read_table(p) for p in sorted((dest/'data'/compact.WAVE/name).glob('*.parquet'))])
                self.assertTrue(restored.equals(expected))

    def test_config_id_mismatch_stops_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=root/'state';state.mkdir();source=root/'source';dest=root/'dest';self.inputs(source,True)
            with patch.multiple(compact,STATE=state,SOURCE=source,DEST=dest),patch.multiple(compact.exact,STATE=state,DEST=dest):
                with self.assertRaisesRegex(ValueError,'unaligned'):compact.job('train')
            self.assertFalse(dest.exists())


if __name__=='__main__':unittest.main()
