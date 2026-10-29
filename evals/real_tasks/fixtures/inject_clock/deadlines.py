import time


def deadline_after(seconds):
    return time.time() + seconds


def is_expired(deadline):
    return time.time() >= deadline
