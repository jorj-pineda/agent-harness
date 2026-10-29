def parse_records(text: str) -> list[list[str]]:
    return [line.split(",") for line in text.splitlines()]
