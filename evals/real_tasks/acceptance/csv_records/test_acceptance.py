import pytest
from records import parse_records


@pytest.mark.parametrize(
    "text,expected",
    [
        ('name,city\nAda,"Paris, France"\n', [["name", "city"], ["Ada", "Paris, France"]]),
        ('"He said ""hi""",ok', [['He said "hi"', "ok"]]),
        ('a,"line one\nline two",z\nb,c,d', [["a", "line one\nline two", "z"], ["b", "c", "d"]]),
        ('a,"one\r\ntwo",z\r\nb,c,d\r\n', [["a", "one\r\ntwo", "z"], ["b", "c", "d"]]),
        ('"one\r\ntwo\nthree",x\n', [["one\r\ntwo\nthree", "x"]]),
    ],
    ids=["quoted-comma", "doubled-quote", "quoted-lf", "quoted-crlf", "mixed-embedded-newlines"],
)
def test_quoted_fields_and_record_boundaries(text, expected):
    assert parse_records(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", []),
        ("\n", [[]]),
        ("\r\n\r\n", [[], []]),
        (',,"",\n', [["", "", "", ""]]),
        ("a,b", [["a", "b"]]),
        ("a,b\n", [["a", "b"]]),
        ("a,b\r\n\r\nc,d", [["a", "b"], [], ["c", "d"]]),
        (' 001 ,"  02  ", 3 \r\n', [[" 001 ", "  02  ", " 3 "]]),
    ],
    ids=[
        "empty-text",
        "blank-record",
        "two-blank-records",
        "empty-fields",
        "no-final-newline",
        "final-newline",
        "interior-blank-record",
        "spaces-and-numeric-strings",
    ],
)
def test_empty_fields_blank_records_and_literal_values(text, expected):
    assert parse_records(text) == expected
