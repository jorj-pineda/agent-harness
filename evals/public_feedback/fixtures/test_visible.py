import pytest
from pagination import first_page, page_items


def test_second_page():
    assert page_items(["a", "b", "c", "d", "e"], 2, 2) == ["c", "d"]


def test_existing_first_page():
    assert first_page([1, 2, 3], 2) == [1, 2]


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("argument", ["page", "page_size"])
@pytest.mark.parametrize("invalid", [True, False])
def test_bool_arguments_are_invalid(empty, argument, invalid):
    values = [] if empty else [1, 2]
    arguments = {"page": 1, "page_size": 2}
    arguments[argument] = invalid
    with pytest.raises(ValueError):
        page_items(values, **arguments)
