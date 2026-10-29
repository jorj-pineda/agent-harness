import pytest
from calc import divide


def test_division() -> None:
    assert divide(6, 2) == 3.0
    assert divide(-9, 3) == -3.0
    assert divide(5, 2) == 2.5
    assert divide(-5, 2) == -2.5
    assert divide(5, -2) == -2.5
    assert divide(-5, -2) == 2.5
    assert divide(1.5, 0.5) == 3.0
    assert divide(0, 2) == 0.0


def test_zero_division() -> None:
    for numerator in (1, 0, -1):
        for denominator in (0, 0.0, -0.0):
            with pytest.raises(ZeroDivisionError):
                divide(numerator, denominator)
