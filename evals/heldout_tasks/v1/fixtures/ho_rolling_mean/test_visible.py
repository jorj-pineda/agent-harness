from rolling import rolling_mean


def test_one_full_window():
    assert rolling_mean([2, 4], 2) == [3]
