from stats import weighted_average


def test_weighted_mean():
    assert weighted_average([(10, 1), (20, 3)]) == 17.5
