from dedupe import dedupe


def test_order() -> None:
    assert dedupe([5, 1, 5]) == [5, 1]
