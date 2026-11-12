import pytest
from lookup import option


@pytest.mark.parametrize("value", [None, False, 0, 0.0, "", [], {}, "green", [1]])
def test_presence_identity_and_preservation(value):
    options = {"setting": value}
    fallback = ["fallback"]
    assert option(options, "setting", fallback) is value
    assert options == {"setting": value}
    assert options["setting"] is value
    assert fallback == ["fallback"]


def test_missing_returns_same_default_without_insertion():
    options = {"other": False}
    fallback = {}
    assert option(options, "absent", fallback) is fallback
    assert options == {"other": False}
    assert fallback == {}
