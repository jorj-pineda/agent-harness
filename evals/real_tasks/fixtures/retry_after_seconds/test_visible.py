from retry_after import parse_retry_after
from retry_client import retry_delay


def test_numeric_retry_header():
    assert parse_retry_after(" 12 ") == 12
    assert retry_delay({"Retry-After": "12"}) == 12


def test_existing_default():
    assert retry_delay({}) == 5
