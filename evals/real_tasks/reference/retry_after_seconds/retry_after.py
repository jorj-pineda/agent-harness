def parse_retry_after(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    digits = value.strip()
    if not digits or not digits.isascii() or not digits.isdecimal():
        return None
    digits = digits.lstrip("0") or "0"
    if len(digits) > 5:
        return None
    seconds = int(digits)
    return seconds if seconds <= 86400 else None
