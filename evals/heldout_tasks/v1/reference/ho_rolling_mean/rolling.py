from collections import deque


def rolling_mean(values, window):
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("Window must be a positive integer")
    recent = deque()
    total = 0
    result = []
    for value in values:
        recent.append(value)
        total += value
        if len(recent) > window:
            total -= recent.popleft()
        if len(recent) == window:
            result.append(total / window)
    return result
