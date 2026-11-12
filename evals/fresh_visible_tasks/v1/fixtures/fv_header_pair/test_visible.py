from headers import parse_header


def test_record_fields():
    pair = parse_header("Name: value")
    assert pair.name == "Name"
    assert pair.value == "value"
    assert pair == ("Name", "value")
