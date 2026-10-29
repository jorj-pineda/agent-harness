def iter_batches(items, size):
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer")
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
