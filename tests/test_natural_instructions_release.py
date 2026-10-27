import sqlite3
import unittest

from scripts.release_natural_instructions_v26_v1 import choose_samples, check_cumulative_index


class NIReleaseTest(unittest.TestCase):
    def test_previous_reviews_are_not_new_samples(self):
        rows = [{'record_id': str(n), 'domain': 'reading',
                 'source_refs': {'answer_selection': {'method': 'minimum_v26_conditional_answer_ppl'}}}
                for n in range(20)]
        a = choose_samples(iter(rows), {'0', '1', '2'}, quota=4)
        b = choose_samples(iter(reversed(rows)), {'0', '1', '2'}, quota=4)
        self.assertEqual([r['record_id'] for r in a], [r['record_id'] for r in b])
        self.assertEqual(len(a), 4)
        self.assertFalse({r['record_id'] for r in a} & {'0', '1', '2'})

    def database(self):
        db = sqlite3.connect(':memory:'); self.addCleanup(db.close)
        schema = '(record_id TEXT,query_hash TEXT,family_id TEXT,split TEXT)'
        db.execute('CREATE TABLE records' + schema)
        db.execute("ATTACH DATABASE ':memory:' AS previous")
        db.execute('CREATE TABLE previous.records' + schema)
        db.execute("INSERT INTO previous.records VALUES ('old','oldq','fam','train')")
        return db

    def test_query_aliases_and_prior_duplicates_fail(self):
        db = self.database()
        db.execute("INSERT INTO records VALUES ('a','q','f','train')")
        db.execute("INSERT INTO records VALUES ('b','q','g','train')")
        with self.assertRaisesRegex(ValueError, 'multiple selected views'):
            check_cumulative_index(db)
        db.execute('DELETE FROM records')
        db.execute("INSERT INTO records VALUES ('a','oldq','f','train')")
        with self.assertRaisesRegex(ValueError, 'published question'):
            check_cumulative_index(db)

    def test_prior_fold_preserved_even_for_different_question(self):
        db = self.database()
        db.execute("INSERT INTO records VALUES ('a','newq','fam','validation')")
        with self.assertRaisesRegex(ValueError, 'published family fold'):
            check_cumulative_index(db)
        db.execute("UPDATE records SET split='train'")
        self.assertEqual(check_cumulative_index(db)['query_hash'], 1)


if __name__ == '__main__': unittest.main()
