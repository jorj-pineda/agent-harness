import copy

import pytest
from windows import merge_windows


@pytest.mark.parametrize(
    "intervals, expected",
    [
        ([], []),
        ([(3, 8)], [(3, 8)]),
        ([(8, 12), (2, 5), (5, 8)], [(2, 12)]),
        ([(0, 10), (3, 4), (0, 10)], [(0, 10)]),
        ([(12, 15), (-5, -1), (-3, 2)], [(-5, 2), (12, 15)]),
        ([(2, 5), (3, 9), (8, 11)], [(2, 11)]),
        ([[4, 9], [1, 4]], [(1, 9)]),
    ],
)
def test_union_and_input_preservation(intervals, expected):
    before = copy.deepcopy(intervals)
    result = merge_windows(intervals)
    assert result == expected
    assert all(type(pair) is tuple for pair in result)
    assert intervals == before
