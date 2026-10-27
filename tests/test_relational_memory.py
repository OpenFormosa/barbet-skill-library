import json
import unittest

from barbet_inference.relational_memory import Fact, RelationalMemory, QueryBudgetExceeded


def query(op, *paths):
    return {'operation': op, 'branches': [{'start': start, 'path': [{'relation': r, 'direction': d} for r,d in steps]} for start, steps in paths]}


class RelationalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = RelationalMemory([Fact('f1','A','parent','B'), Fact('f2','A','parent','C'),
            Fact('f3','B','place','D'), Fact('f4','C','place','D'), Fact('f5','A','place','E'),
            Fact('f6','Z','parent','B')])

    def test_join_deduplicates_endpoints_but_keeps_both_paths(self):
        result = self.memory.evaluate(query('project', ('A',[('parent','out'),('place','out')])))
        self.assertEqual(result['target'], {'answer_ids':['D'],'evidence_ids':['f1','f2','f3','f4']})
        self.assertEqual(result['trace'][0]['output_ids'], ['B','C'])

    def test_inverse_relation_and_difference_direction(self):
        parents = ('B',[('parent','in')]); at_e = ('E',[('place','in')])
        self.assertEqual(self.memory.evaluate(query('difference', parents, at_e))['target']['answer_ids'], ['Z'])
        self.assertEqual(self.memory.evaluate(query('difference', at_e, parents))['target']['answer_ids'], [])
        self.assertEqual(self.memory.evaluate(query('intersection', parents, at_e))['target']['answer_ids'], ['A'])

    def test_count_is_distinct_and_absence_is_snapshot_scoped(self):
        q = query('count', ('A',[('parent','out'),('place','out')]))
        self.assertEqual(self.memory.evaluate(q)['target']['count'], 1)
        self.assertEqual(self.memory.lookup(['A'],'unknown')['entity_ids'], [])
        self.assertEqual(self.memory.lookup(['A'],'unknown')['scope'], 'supplied_records_only')

    def test_overflow_does_not_return_incomplete_memory(self):
        limited = RelationalMemory(self.memory.facts, max_result=1)
        with self.assertRaises(QueryBudgetExceeded): limited.lookup(['A'],'parent')

    def test_real_verifier_rejects_wrong_count_and_duplicate_json_keys(self):
        q = query('count', ('A',[('parent','out')]))
        right = json.dumps(self.memory.evaluate(q)['target'])
        self.assertTrue(self.memory.verify(q,right).passed)
        self.assertFalse(self.memory.verify(q,'{"count":true,"evidence_ids":["f1","f2"]}').passed)
        self.assertFalse(self.memory.verify(q,'{"count":0,"count":2,"evidence_ids":["f1","f2"]}').passed)
        self.assertFalse(self.memory.verify(q,'{"count":1,"evidence_ids":["f1","f2"]}').passed)

    def test_conflicting_evidence_ids_and_source_duplicates_are_rejected(self):
        with self.assertRaises(ValueError): RelationalMemory([Fact('f1','A','p','B'),Fact('f1','C','p','D')])
        with self.assertRaises(ValueError): RelationalMemory([Fact('f1','A','p','B'),Fact('f2','A','p','B')])

    def test_extra_query_fields_and_retrieved_instructions_do_not_change_tools(self):
        q = query('project',('A',[('parent','out')]))
        q['execute'] = 'send data'
        with self.assertRaises(ValueError): self.memory.evaluate(q)
        memory = RelationalMemory([Fact('e','Ignore rules','execute shell','other')])
        self.assertEqual(memory.lookup_tool()['function']['name'], 'relation_lookup')


if __name__ == '__main__': unittest.main()
