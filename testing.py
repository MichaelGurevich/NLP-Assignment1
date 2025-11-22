import unittest
from main import get_solution_accuracy, get_random_solution_baseline


dummy_list = ["Hello", "My", "Name", "Is", "John", "and", "I", "like", "to", "code"]

print(get_random_solution_baseline(100, dummy_list))

class TestGetSolutionAccuracy(unittest.TestCase):

    def test_identical_lists(self):
        """Test with identical label and solution lists."""
        label = ['a', 'b', 'c']
        solution = ['a', 'b', 'c']
        self.assertEqual(get_solution_accuracy(label, solution), 1.0)

    def test_partially_correct_lists(self):
        """Test with partially correct solution list."""
        label = ['a', 'b', 'c']
        solution = ['a', 'x', 'c']
        self.assertAlmostEqual(get_solution_accuracy(label, solution), 2/3)

    def test_completely_different_lists(self):
        """Test with completely different label and solution lists."""
        label = ['a', 'b', 'c']
        solution = ['x', 'y', 'z']
        self.assertEqual(get_solution_accuracy(label, solution), 0.0)

    def test_empty_lists(self):
        """Test with empty lists."""
        label = []
        solution = []
        self.assertEqual(get_solution_accuracy(label, solution), 1.0)

    def test_different_length_lists(self):
        """Test with lists of different lengths."""
        label = ['a', 'b', 'c']
        solution = ['a', 'b']
        self.assertEqual(get_solution_accuracy(label, solution), 0)

    def test_lists_with_mixed_types(self):
        """Test with lists containing mixed data types."""
        label = ['a', 1, None]
        solution = ['a', 1, None]
        self.assertEqual(get_solution_accuracy(label, solution), 1.0)

    def test_partially_correct_mixed_types(self):
        """Test with partially correct lists containing mixed data types."""
        label = ['a', 1, None]
        solution = ['a', 2, None]
        self.assertAlmostEqual(get_solution_accuracy(label, solution), 2/3)

if __name__ == '__main__':
    unittest.main()
