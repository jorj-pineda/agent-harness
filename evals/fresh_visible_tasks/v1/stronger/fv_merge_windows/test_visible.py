from windows import merge_windows


def test_overlap():
    assert merge_windows([(1, 4), (3, 6)]) == [(1, 6)]


def test_touching_and_nested():
    intervals = [(4, 7), (1, 4), (2, 3)]
    assert merge_windows(intervals) == [(1, 7)]
    assert intervals == [(4, 7), (1, 4), (2, 3)]


def test_empty_and_separate():
    assert merge_windows([]) == []
    assert merge_windows([(9, 10), (1, 2)]) == [(1, 2), (9, 10)]
