from windows import merge_windows


def test_overlap():
    assert merge_windows([(1, 4), (3, 6)]) == [(1, 6)]
