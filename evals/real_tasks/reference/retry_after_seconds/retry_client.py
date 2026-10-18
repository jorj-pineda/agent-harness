from retry_after import parse_retry_after


def retry_delay(headers: dict[str, object], default_seconds: int = 5) -> int:
    header = headers.get("Retry-After", headers.get("retry-after"))
    parsed = parse_retry_after(header)
    return default_seconds if parsed is None else parsed
