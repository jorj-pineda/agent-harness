import endpoint
import pytest


@pytest.mark.parametrize(
    "text,expected",
    [
        ("localhost:80", ("localhost", 80)),
        (":0", ("", 0)),
        (" a : +080 ", (" a ", 80)),
        ("alpha:beta:-2", ("alpha:beta", -2)),
    ],
)
def test_record_preserves_existing_split_and_conversion(text, expected):
    record = endpoint.parse_endpoint(text)
    assert isinstance(record, endpoint.Endpoint) and isinstance(record, tuple)
    assert record == expected and hash(record) == hash(expected)
    host, port = record
    assert len(record) == 2 and record[0] == host == record.host
    assert record[1] == port == record.port


def test_constructor_and_immutability():
    record = endpoint.Endpoint("host", 7)
    assert record == ("host", 7)
    for name in ("host", "port", "extra"):
        with pytest.raises(AttributeError):
            setattr(record, name, "changed")


@pytest.mark.parametrize(
    "text,error",
    [
        ("missing-colon", ValueError),
        ("host:", ValueError),
        ("host:bad", ValueError),
        (None, AttributeError),
        (3, AttributeError),
    ],
)
def test_existing_errors_remain(text, error):
    with pytest.raises(error):
        endpoint.parse_endpoint(text)
