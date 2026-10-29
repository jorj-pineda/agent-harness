import pytest
from stats import weighted_average


@pytest.mark.parametrize(
    "samples,expected",
    [
        ([(10, 1), (20, 3)], 17.5),
        ([(99, 0), (4, 2)], 4),
        ([(-10, 0.5), (2, 1.5)], -1),
        ([(0, 10)], 0),
    ],
)
def test_weighted_mean_single_pass_and_no_mutation(samples, expected):
    before = samples.copy()
    assert weighted_average(samples) == pytest.approx(expected)
    assert weighted_average(iter(samples)) == pytest.approx(expected)
    assert samples == before


@pytest.mark.parametrize(
    "samples", [[], [(1, 0)], [(1, 0), (2, 0)], [(10, -1), (20, 3)], [(5, -2)]]
)
def test_empty_zero_and_negative_weights_raise_value_error(samples):
    with pytest.raises(ValueError):
        weighted_average(iter(samples))
