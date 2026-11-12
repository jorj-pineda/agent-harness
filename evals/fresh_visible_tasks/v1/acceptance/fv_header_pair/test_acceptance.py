import pytest
from headers import HeaderPair, parse_header


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Q: z", ("Q", "z")),
        ("  Key\t:  left:right:tail  ", ("Key", "left:right:tail")),
        (": blank", ("", "blank")),
        ("Empty:", ("Empty", "")),
        (" : ", ("", "")),
    ],
)
def test_preserved_parse_and_record(text, expected):
    result = parse_header(text)
    assert isinstance(result, HeaderPair)
    assert isinstance(result, tuple)
    assert result == expected
    assert (result.name, result.value) == expected
    assert len(result) == 2
    name, value = result
    assert (name, value) == expected
    assert result[0] == expected[0] and result[1] == expected[1]
    assert hash(result) == hash(expected)
    for attribute in ("name", "value"):
        with pytest.raises(AttributeError):
            setattr(result, attribute, "changed")


def test_constructor_and_existing_error():
    assert HeaderPair("a", "b") == ("a", "b")
    with pytest.raises(ValueError):
        parse_header("no colon here")
