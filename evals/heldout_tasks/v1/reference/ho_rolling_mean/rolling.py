from collections import deque
from statistics import mean


def rolling_mean(values, window):
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("Window must be a positive integer")
    recent = deque()
    result = []
    for value in values:
        recent.append(value)
        if len(recent) > window:
            recent.popleft()
        if len(recent) == window:
            result.append(mean(recent))
    return result
