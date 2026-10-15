"""Authored development checks; never supplied to the repair model."""
from .data import digest


CASES = {
    'gcd': '''
import math
def test_grid():
    for a in range(18):
        for b in range(18):
            assert target.gcd(a, b) == math.gcd(a, b)
''',
    'find_in_sorted': '''
def test_boundaries_duplicates_and_absence():
    for values in [[], [2], [-3, -1, 0, 4], [1, 1, 1, 3, 3], list(range(0, 30, 2))]:
        for key in range(-5, 33):
            index = target.find_in_sorted(values[:], key)
            if key in values:
                assert isinstance(index, int) and 0 <= index < len(values) and values[index] == key
            else:
                assert index == -1
''',
    'quicksort': '''
from itertools import product
def test_small_multisets():
    for size in range(5):
        for items in product([-1, 0, 1], repeat=size):
            assert target.quicksort(list(items)) == sorted(items)
''',
    'flatten': '''
def test_nested_objects():
    marker = object()
    assert list(target.flatten([])) == []
    assert list(target.flatten([[], [[], []]])) == []
    assert list(target.flatten([0, [False, [None, 'ab']], (1, 2), [[marker]]])) == [0, False, None, 'ab', (1, 2), marker]
''',
    'kheapsort': '''
def test_valid_displacements():
    for size in range(1, 13):
        ordered = list(range(size))
        assert list(target.kheapsort(ordered[:], 0)) == ordered
        assert list(target.kheapsort(ordered[::-1], size - 1)) == ordered
        pairs = ordered[:]
        for i in range(0, size - 1, 2):
            pairs[i], pairs[i + 1] = pairs[i + 1], pairs[i]
        assert list(target.kheapsort(pairs, 1)) == ordered
''',
    'knapsack': '''
from itertools import combinations
def test_small_subset_oracle():
    for items in [[], [(1, 7)], [(2, 5), (2, 6), (3, 9)], [(1, 2), (4, 8), (2, 5), (6, 13)]]:
        for capacity in range(10):
            expected = max([0] + [sum(v for w, v in subset) for n in range(len(items) + 1)
                for subset in combinations(items, n) if sum(w for w, v in subset) <= capacity])
            assert target.knapsack(capacity, items[:]) == expected
''',
    'sqrt': '''
import math
def test_positive_domain():
    for value in [1.0, 1.25, 2.0, 9.0, 31.0, 121.0, 10000.0]:
        for tolerance in [0.001, 0.01, 0.1]:
            assert abs(target.sqrt(value, tolerance) - math.sqrt(value)) <= tolerance
''',
    'shunting_yard': '''
def test_precedence_and_left_association():
    assert target.shunting_yard([8]) == [8]
    assert target.shunting_yard([8, '-', 3, '-', 1]) == [8, 3, '-', 1, '-']
    assert target.shunting_yard([8, '/', 2, '*', 3]) == [8, 2, '/', 3, '*']
    assert target.shunting_yard([2, '+', 3, '*', 4, '-', 5]) == [2, 3, 4, '*', '+', 5, '-']
''',
}


def suite(name):
    if name not in CASES:
        return None
    return ('import importlib\nimport pytest\n'
            f"target = importlib.import_module(('correct_python_programs.' if pytest.use_correct else 'python_programs.') + '{name}')\n"
            + CASES[name]).encode()


def manifest():
    return {name: digest(suite(name)) for name in sorted(CASES)}


def run_regressions(task, runner, *, reference=False, source=None):
    code = suite(task.name)
    if code is None:
        return None
    files = dict(task.files)
    if source is not None:
        files[f'python_programs/{task.name}.py'] = source.encode()
    path = 'python_testcases/test_authored_regression.py'
    files[path] = code
    return runner.run(files, path, reference=reference)
