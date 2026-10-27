import unittest

from barbet_inference.sft import Rejected
from scripts.natural_instructions_memory_adapter import adapt, exact_openorca_views


class NaturalInstructionsMemoryTests(unittest.TestCase):
    def task(self):
        return {'Definition': ['Return the requested code from the supplied record.'],
                'Categories': ['Information Extraction'], 'Instance License': ['CC BY 4.0'],
                'Positive Examples': [
                    {'input': 'prior record', 'output': 'PRIOR-EXAMPLE'},
                    {'input': 'target record', 'output': 'SECRET-TARGET'},
                    {'input': 'another record', 'output': 'OTHER-EXAMPLE'}]}

    def test_target_answer_never_enters_memory_and_target_demonstration_is_removed(self):
        r = adapt(self.task(), {'id': 'x', 'input': 'target record', 'output': ['SECRET-TARGET', 'SECOND-TARGET']}, source_url='source')
        self.assertNotIn('SECRET-TARGET', r['prompt'])
        self.assertNotIn('SECOND-TARGET', r['prompt'])
        self.assertEqual(r['target_or_duplicate_example_indices_excluded'], [1])
        self.assertEqual(r['worked_example_indices'], [0, 2])
        self.assertEqual(r['memory']['documents'][-1]['text'], 'target record')

    def test_answer_versions_share_the_same_input_and_memory_position(self):
        a = adapt(self.task(), {'input': 'target record', 'output': ['FIRST']}, source_url='source')
        b = adapt(self.task(), {'input': 'target record', 'output': ['SECOND', 'FIRST', 'SECOND']}, source_url='source')
        self.assertEqual(a['prompt'], b['prompt'])
        self.assertEqual(a['source_task_key'], b['source_task_key'])
        self.assertEqual(b['answers'], ['SECOND', 'FIRST'])

    def test_unknown_instance_license_is_not_covered_by_repository_apache_license(self):
        task = self.task(); task['Instance License'] = ['Unknown']
        with self.assertRaisesRegex(Rejected, 'license'):
            adapt(task, {'input': 'record', 'output': ['answer']}, source_url='source')

    def test_multiple_definitions_or_wrong_answer_generation_are_not_silently_selected(self):
        task = self.task(); task['Definition'].append('Conflicting task')
        with self.assertRaisesRegex(Rejected, 'ambiguous'):
            adapt(task, {'input': 'record', 'output': ['answer']}, source_url='source')
        task = self.task(); task['Categories'] = ['Wrong Candidate Generation']
        with self.assertRaisesRegex(Rejected, 'category'):
            adapt(task, {'input': 'record', 'output': ['answer']}, source_url='source')

    def test_wrapper_keys_retain_the_actual_definition_and_input(self):
        views = exact_openorca_views('D', 'I')
        self.assertIn('Definition: D\nInput: I\nOutput:', views)
        self.assertEqual(len(views), len(set(views)))


if __name__ == '__main__':
    unittest.main()
