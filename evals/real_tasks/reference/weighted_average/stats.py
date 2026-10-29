def weighted_average(samples):
    numerator = 0
    denominator = 0
    for value, weight in samples:
        if weight < 0:
            raise ValueError("Negative weight")
        numerator += value * weight
        denominator += weight
    if denominator == 0:
        raise ValueError("No positive weight")
    return numerator / denominator
