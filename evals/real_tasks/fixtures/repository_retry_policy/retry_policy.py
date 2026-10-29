from retry_config import DEFAULT_BASE_DELAY, DEFAULT_MAX_ATTEMPTS, MAX_DELAY


def max_attempts(config):
    return max(1, int(config.get("max_attempts", DEFAULT_MAX_ATTEMPTS)))


def delay_seconds(failure_index, config):
    base = config.get("base_delay", DEFAULT_BASE_DELAY)
    return min(MAX_DELAY, base * 2 ** max(0, failure_index - 1))
