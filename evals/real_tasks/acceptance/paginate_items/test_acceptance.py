import pytest
from pagination import first_page, page_items


@pytest.mark.parametrize(
    "page,size,expected",
    [(1, 2, [0, 1]), (2, 2, [2, 3]), (3, 2, [4]), (4, 2, []), (1, 99, list(range(5)))],
)
def test_one_based_boundaries(page, size, expected):
    assert page_items(list(range(5)), page, size) == expected


def test_empty_and_element_identity_without_input_mutation():
    assert page_items([], 1, 1) == []
    marker = object()
    values = [marker, {"nested": [1]}, marker]
    original = values.copy()
    result = page_items(values, 1, 20)
    assert result == original and result is not values
    assert all(a is b for a, b in zip(result, original, strict=True))
    result.clear()
    assert values == original


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "2", None])
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_arguments_even_for_empty_input(bad, empty):
    values = [] if empty else [1, 2]
    with pytest.raises(ValueError):
        page_items(values, bad, 2)
    with pytest.raises(ValueError):
        page_items(values, 1, bad)


def test_existing_first_page_behavior_is_preserved():
    values = [1, 2, 3]
    for size in [0, 2, -1, 99]:
        assert first_page(values, size) == values[:size]
