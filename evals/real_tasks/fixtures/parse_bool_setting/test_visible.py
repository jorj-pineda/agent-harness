from bool_settings import parse_bool


def test_boolean_tokens():
    assert parse_bool("YES") is True
    assert parse_bool("off") is False
