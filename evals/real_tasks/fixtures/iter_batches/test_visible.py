from batching import iter_batches


def test_batches():
    assert list(iter_batches([1, 2, 3], 2)) == [[1, 2], [3]]
