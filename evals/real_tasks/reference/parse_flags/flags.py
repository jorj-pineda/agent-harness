def parse_flags(raw: str) -> dict[str, str]:
    flags: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        flags[key.strip()] = value.strip()
    return flags
