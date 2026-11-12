from typing import NamedTuple


class HeaderPair(NamedTuple):
    name: str
    value: str


def parse_header(text):
    name, value = text.split(":", 1)
    return HeaderPair(name.strip(), value.strip())
