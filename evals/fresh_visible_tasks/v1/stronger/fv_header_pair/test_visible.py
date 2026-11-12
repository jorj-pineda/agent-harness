import pytest
from headers import parse_header


def test_record_fields():
    pair = parse_header("Name: value")
    assert pair.name == "Name"
    assert pair.value == "value"
    assert pair == ("Name", "value")


def test_first_colon_empty_fields_and_immutable():
    assert parse_header(" X : a:b ") == ("X", "a:b")
    assert parse_header(":") == ("", "")
    with pytest.raises(AttributeError):
        parse_header("A: b").name = "changed"


def test_missing_colon():
    with pytest.raises(ValueError):
        parse_header("missing")
