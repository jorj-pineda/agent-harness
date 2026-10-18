from pagination import first_page, page_items


def test_second_page():
    assert page_items(["a", "b", "c", "d", "e"], 2, 2) == ["c", "d"]


def test_existing_first_page():
    assert first_page([1, 2, 3], 2) == [1, 2]
