import time


def deadline_after(seconds, *, clock=None):
    now = time.time() if clock is None else clock()
    return now + seconds


def is_expired(deadline, *, clock=None):
    now = time.time() if clock is None else clock()
    return now >= deadline
