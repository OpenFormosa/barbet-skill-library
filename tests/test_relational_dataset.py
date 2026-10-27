from dataclasses import asdict
import copy
import json
import unittest

from barbet_inference.relational_memory import Fact,RelationalMemory
from barbet_inference.relational_dataset import restore_relational_memory,relation_tool_callbacks


class RelationalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.memory=RelationalMemory([Fact('f1','A','p','B')])
        self.metadata={'registered_relation_snapshot':{'format':'barbet-finite-relations-v1',
            'tool_name':'relation_lookup','visibility':'registered_tool_state; observations become model-visible only after calls',
            'snapshot_sha256':self.memory.snapshot_sha256,'facts':[asdict(f) for f in self.memory.facts]}}

    def test_portable_dataset_field_restores_actual_callback(self):
        row={'memory_json':json.dumps(self.metadata)}
        callback=relation_tool_callbacks(row)['relation_lookup']
        self.assertEqual(callback(['A'],'p'),self.memory.lookup(['A'],'p'))

    def test_changed_fact_cannot_reuse_old_snapshot_proof(self):
        metadata=copy.deepcopy(self.metadata);metadata['registered_relation_snapshot']['facts'][0]['object']='C'
        with self.assertRaises(ValueError):restore_relational_memory({'memory':metadata})

    def test_metadata_does_not_register_arbitrary_tool(self):
        metadata=copy.deepcopy(self.metadata);metadata['registered_relation_snapshot']['tool_name']='shell'
        with self.assertRaises(ValueError):relation_tool_callbacks({'memory':metadata})


if __name__=='__main__':unittest.main()
