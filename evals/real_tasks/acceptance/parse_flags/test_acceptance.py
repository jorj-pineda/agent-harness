from flags import parse_flags


def test_comments_and_values() -> None:
    assert parse_flags("# note\n A = one=two \n\nB=three\n") == {
        "A": "one=two",
        "B": "three",
    }
    assert parse_flags("\n # only a comment\n") == {}
