import endpoint


def test_named_record():
    record = endpoint.parse_endpoint("localhost:80")
    assert isinstance(record, endpoint.Endpoint)
    assert record.host == "localhost" and record.port == 80
