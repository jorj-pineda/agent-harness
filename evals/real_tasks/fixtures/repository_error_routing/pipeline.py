from client import fetch
from errors import PermanentError, TransientError


def handle(record):
    try:
        payload = fetch(record)
    except TransientError:
        return {"state": "retry", "value": None}
    except PermanentError:
        return {"state": "failed", "value": None}
    if payload is None:
        return {"state": "empty", "value": None}
    return {"state": "ok", "value": payload}
