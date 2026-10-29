def weighted_average(samples):
    return sum(value * weight for value, weight in samples) / len(samples)
