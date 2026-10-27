import unittest
from fractions import Fraction
from barbet_inference.arithmetic_verifier import evaluate_arithmetic, verify_integer_calculation, UnsupportedArithmetic


class ArithmeticVerifierTests(unittest.TestCase):
    def test_source_high_ppl_answer_is_correct(self):
        q = r'Calculate the value of \((-2)^{3} - (-3)^{2}\). Provide your answer as a single integer.'
        self.assertEqual(verify_integer_calculation(q, '-17')['status'], 'verified')
        self.assertEqual(verify_integer_calculation(q, '1')['status'], 'incorrect')

    def test_precedence_and_exact_fractions(self):
        cases = {'-2^2': Fraction(-4), '(-2)^2': Fraction(4), '2^3^2': Fraction(512),
                 '2^-3': Fraction(1, 8), r'\frac{3}{2} * 4': Fraction(6), '0.1 + 0.2': Fraction(3, 10)}
        for text, expected in cases.items(): self.assertEqual(evaluate_arithmetic(text), expected)

    def test_names_code_and_unbounded_operations_are_not_executed(self):
        for text in ['__import__("os").system("echo unsafe")', 'sin(0)', '2^999999', '0^-1', '1/0', '(1+2', '2 3']:
            with self.assertRaises(UnsupportedArithmetic): evaluate_arithmetic(text)

    def test_additional_conditions_and_long_responses_are_outside_scope(self):
        q = 'Compute 2 + 3. Provide your answer as a single integer.'
        self.assertEqual(verify_integer_calculation(q, '5')['status'], 'verified')
        self.assertEqual(verify_integer_calculation(q + ' Use modulus 3.', '2')['status'], 'outside_scope')
        self.assertEqual(verify_integer_calculation(q, 'The sum is 5.')['status'], 'outside_scope')


if __name__ == '__main__': unittest.main()
