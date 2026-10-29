from errors import PermanentError, TransientError


def fetch(record):
    if record["kind"] == "transient":
        raise TransientError("try later")
    if record["kind"] == "permanent":
        raise PermanentError("not available")
    return record["payload"]
