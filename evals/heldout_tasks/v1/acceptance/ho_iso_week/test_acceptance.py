from datetime import date

import pytest
from week_key import week_key


@pytest.mark.parametrize(
    "day,expected",
    [
        (date(2016, 1, 1), "2015-W53"),
        (date(2019, 12, 30), "2020-W01"),
        (date(2020, 12, 31), "2020-W53"),
        (date(2021, 1, 1), "2020-W53"),
        (date(2021, 1, 4), "2021-W01"),
        (date(2024, 12, 30), "2025-W01"),
        (date(1, 1, 1), "0001-W01"),
        (date(2024, 2, 29), "2024-W09"),
    ],
)
def test_iso_year_week_and_padding(day, expected):
    before = day.isoformat()
    assert week_key(day) == expected
    assert day.isoformat() == before
