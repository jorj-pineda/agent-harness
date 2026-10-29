import pytest
from batching import iter_batches


@pytest.mark.parametrize("size", [1, 2, 5, 99])
def test_order_short_tail_identity_and_preservation(size):
    values = [object(), {}, [], object(), None]
    before = values.copy()
    batches = list(iter_batches(values, size))
    assert [len(batch) for batch in batches] == [
        len(values[i : i + size]) for i in range(0, len(values), size)
    ]
    flattened = [item for batch in batches for item in batch]
    assert all(a is b for a, b in zip(flattened, values, strict=True))
    assert len({id(batch) for batch in batches}) == len(batches)
    assert values == before
    assert values[1] == {} and values[2] == []
    assert list(iter_batches([], size)) == []


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "2", None])
@pytest.mark.parametrize("values", [[], [1]])
def test_invalid_sizes_even_when_empty(bad, values):
    batches = iter_batches(iter(values), bad)
    with pytest.raises(ValueError):
        next(iter(batches))


def test_single_pass_laziness_and_no_batch_prefetch():
    seen = []

    def source():
        for value in range(5):
            seen.append(value)
            yield value

    batches = iter(iter_batches(source(), 2))
    assert seen == []
    assert next(batches) == [0, 1] and seen == [0, 1]
    assert next(batches) == [2, 3] and seen == [0, 1, 2, 3]
    assert next(batches) == [4] and seen == [0, 1, 2, 3, 4]
    with pytest.raises(StopIteration):
        next(batches)
