def parse_flags(raw: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in raw.splitlines())
