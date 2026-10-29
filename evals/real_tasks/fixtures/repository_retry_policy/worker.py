from retry_policy import max_attempts


def should_retry(failed_attempt, config):
    return failed_attempt < max_attempts(config)
