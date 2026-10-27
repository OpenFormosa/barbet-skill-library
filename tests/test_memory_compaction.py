import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
from scripts import compact_memory_release_v2 as compact


class CompactionTests(unittest.TestCase):
    def test_hold_filter_and_shard_boundary_preserve_exact_token_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';dest=root/'dest';state=root/'state';state.mkdir()
            (state/'postpublication-tool-interface-audit-v2.json').write_text(json.dumps({'record_ids':['b']}))
            records=[]
            for rid in ['a','b','c','d']:
                records.append({'record_id':rid,'family_id':rid,'query_hash':rid,'split':'train','domain':'reading',
                    'source':'source','upstream_source':'source','license':'mit','verification_tier':'fixture',
                    'memory_placement':'front','observation_origin':'fixture','actual_retrieval':False,'recorded_error_followup':False,
                    'pangolin_tokens':3,'pangolin_supervised_tokens':2,'minicpm_tokens':3,'minicpm_supervised_tokens':2,
                    'assistant_turns':1,'tool_observations':0,'text':'full text '+rid})
            for name in ['records','tokenized','minicpm5_tokenized']:
                p=source/'data/wave-0001'/name/'train-00000.parquet';p.parent.mkdir(parents=True)
                values=records if name=='records' else [{'record_id':r,'input_ids':[1,n,3],'labels':[-100,n,3]} for n,r in enumerate(['a','b','c','d'])]
                pq.write_table(pa.Table.from_pylist(values),p)
            before={str(p):p.read_bytes() for p in source.rglob('*.parquet')}
            with patch.object(compact,'ROOTS',{'wave-0001':source}),patch.object(compact,'DEST',dest),patch.object(compact,'STATE',state),patch.object(compact,'FIRST',state),patch.object(compact,'ROWS',2):
                result=compact.job(('wave-0001','train'))
            self.assertEqual(result['counts']['accepted'],3)
            self.assertEqual(result['counts']['excluded:new_declared_argument_guard'],1)
            tables=[pq.read_table(p) for p in sorted((dest/'data/wave-0001/tokenized').glob('*.parquet'))]
            self.assertEqual([t.num_rows for t in tables],[2,1])
            self.assertEqual(pa.concat_tables(tables).to_pylist(),[
                {'record_id':'a','input_ids':[1,0,3],'labels':[-100,0,3]},
                {'record_id':'c','input_ids':[1,2,3],'labels':[-100,2,3]},
                {'record_id':'d','input_ids':[1,3,3],'labels':[-100,3,3]}])
            for path,content in before.items():self.assertEqual(Path(path).read_bytes(),content)

    def test_conflicting_existing_shard_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'out.parquet';pq.write_table(pa.table({'x':[1]}),p);before=p.read_bytes()
            with patch.object(compact,'DEST',Path(folder)),self.assertRaisesRegex(ValueError,'conflict'):
                compact.write_checked(pa.table({'x':[2]}),p,wave='w',config='records',split='train')
            self.assertEqual(p.read_bytes(),before)


if __name__=='__main__':unittest.main()
