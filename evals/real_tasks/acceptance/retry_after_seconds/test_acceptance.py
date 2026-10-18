import pytest
from retry_after import parse_retry_after
from retry_client import retry_delay


@pytest.mark.parametrize(
    "value,expected",
    [("0", 0), (" 002 \t", 2), ("86400", 86400), ("0" * 5000, 0), ("0" * 5000 + "1", 1)],
    ids=["zero", "padded-two", "upper-bound", "leading-zeros-5000", "leading-zeros-5000-plus-one"],
)
def test_numeric_boundaries_and_leading_zeros(value, expected):
    parsed = parse_retry_after(value)
    delay = retry_delay({"Retry-After": value}, 91)
    assert type(parsed) is int and parsed == expected
    assert type(delay) is int and delay == expected


@pytest.mark.parametrize(
    "value",
    [
        None,
        12,
        True,
        [],
        "",
        " \n",
        "-1",
        "+1",
        "1.0",
        "1e2",
        "1_000",
        "١٢",
        "\uff11\uff12",  # Fullwidth digits are intentionally unsupported.
        "Wed, 21 Oct 2015 07:28:00 GMT",
        "86401",
        "9" * 5000,
    ],
    ids=[
        "none",
        "integer",
        "boolean",
        "list",
        "empty",
        "whitespace",
        "negative",
        "signed",
        "fraction",
        "exponent",
        "underscore",
        "arabic-indic-digits",
        "fullwidth-digits",
        "http-date",
        "over-bound",
        "oversized-digits-5000",
    ],
)
def test_unsupported_inputs_fall_back_without_exceptions(value):
    assert parse_retry_after(value) is None
    assert retry_delay({"Retry-After": value}, 91) == 91


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({}, 91),
        ({"retry-after": "7"}, 7),
        ({"RETRY-AFTER": "7"}, 91),
        ({"Retry-After": "0", "retry-after": "7"}, 0),
        ({"Retry-After": "invalid", "retry-after": "7"}, 91),
        ({"Retry-After": None, "retry-after": "7"}, 91),
    ],
)
def test_header_precedence_and_no_mutation(headers, expected):
    before = headers.copy()
    assert retry_delay(headers, 91) == expected
    assert headers == before


def test_existing_default_is_returned_unchanged():
    marker = object()
    assert retry_delay({}, marker) is marker
    assert retry_delay({"Retry-After": "not numeric"}, marker) is marker
    assert retry_delay({}) == 5
