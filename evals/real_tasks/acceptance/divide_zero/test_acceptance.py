import pytest
from calc import divide


def test_division() -> None:
    assert divide(6, 2) == 3.0
    assert divide(-9, 3) == -3.0


def test_zero_division() -> None:
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
