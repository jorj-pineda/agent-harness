from dedupe import dedupe


def test_order_and_duplicates() -> None:
    assert dedupe([5, 1, 5, 3, 1]) == [5, 1, 3]
    assert dedupe([]) == []
    assert dedupe([0, -1, 0]) == [0, -1]
