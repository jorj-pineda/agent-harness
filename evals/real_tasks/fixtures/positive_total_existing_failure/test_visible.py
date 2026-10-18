from totals import total_positive


def test_positive_only():
    assert total_positive([1, -2, 3]) == 4
