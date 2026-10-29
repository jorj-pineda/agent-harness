from itertools import product

import pytest
from rolling import rolling_mean


def test_all_full_windows_and_input_preservation():
    values = [0, 1, 2, 3]
    before = values.copy()
    assert rolling_mean(iter(values), 2) == [0.5, 1.5, 2.5]
    assert values == before
    assert rolling_mean([1.5, -0.5, 2.0], 2) == [0.5, 0.75]
    assert rolling_mean([], 1) == []
    assert rolling_mean([1], 2) == []


def test_small_finite_windows_against_direct_formula():
    for length in range(5):
        for values in product((-2, 0, 3), repeat=length):
            for window in range(1, 6):
                expected = [
                    sum(values[i : i + window]) / window for i in range(length - window + 1)
                ]
                assert rolling_mean(iter(values), window) == pytest.approx(expected)


def test_finite_magnitudes_do_not_poison_later_windows():
    assert rolling_mean([10**16, 1.0, 1.0], 2) == [5e15, 1.0]
    assert rolling_mean([1e308, 1e308], 2) == [1e308]


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "2", None])
def test_invalid_window_precedes_any_input_read(bad):
    def unread():
        pytest.fail("Invalid window consumed input")
        yield 1

    with pytest.raises(ValueError):
        rolling_mean(unread(), bad)
    with pytest.raises(ValueError):
        rolling_mean([], bad)
