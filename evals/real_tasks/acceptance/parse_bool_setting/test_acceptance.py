import pytest
from bool_settings import parse_bool


@pytest.mark.parametrize("text", ["true", "YES", "On", "1", " \tTrUe\n"])
def test_true_tokens(text):
    assert parse_bool(text) is True


@pytest.mark.parametrize("text", ["false", "NO", "Off", "0", "\t false \n"])
def test_false_tokens(text):
    assert parse_bool(text, True) is False


@pytest.mark.parametrize(
    "text", ["", " \t", "enabled", "2", "yes please", True, False, 1, 0, [], {}]
)
def test_other_inputs_raise_value_error(text):
    with pytest.raises(ValueError):
        parse_bool(text)


def test_none_returns_default_without_coercion():
    marker = object()
    assert parse_bool(None) is False
    assert parse_bool(None, marker) is marker
