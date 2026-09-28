from flags import parse_flags


def test_comment() -> None:
    assert parse_flags("# note\n A = B\n") == {"A": "B"}
