from records import parse_records


def test_quoted_comma():
    assert parse_records('name,city\nAda,"Paris, France"\n') == [
        ["name", "city"],
        ["Ada", "Paris, France"],
    ]
