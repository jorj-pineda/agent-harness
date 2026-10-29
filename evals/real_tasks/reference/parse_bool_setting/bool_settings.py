def parse_bool(text, default=False):
    if text is None:
        return default
    if not isinstance(text, str):
        raise ValueError("Expected a boolean string")
    token = text.strip().lower()
    if token in {"true", "yes", "on", "1"}:
        return True
    if token in {"false", "no", "off", "0"}:
        return False
    raise ValueError("Unknown boolean token")
