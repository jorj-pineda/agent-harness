import pytest
from lookup import option


def test_present_and_missing():
    assert option({"color": "blue"}, "color", "red") == "blue"
    assert option({}, "color", "red") == "red"


@pytest.mark.parametrize("value", [None, False, 0, "", [], {}])
def test_falsey_present_values(value):
    assert option({"present": value}, "present", object()) is value
