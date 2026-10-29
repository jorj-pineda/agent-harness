from datetime import date

from week_key import week_key


def test_iso_year_boundary():
    assert week_key(date(2021, 1, 1)) == "2020-W53"
