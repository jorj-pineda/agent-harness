from typing import NamedTuple


class Endpoint(NamedTuple):
    host: str
    port: int


def parse_endpoint(text):
    host, port = text.rsplit(":", 1)
    return Endpoint(host, int(port))
