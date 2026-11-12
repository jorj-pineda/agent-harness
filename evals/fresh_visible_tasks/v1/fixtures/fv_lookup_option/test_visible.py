from lookup import option


def test_present_and_missing():
    assert option({"color": "blue"}, "color", "red") == "blue"
    assert option({}, "color", "red") == "red"
