from calc import add, divide


def test_add() -> None:
    assert add(1, 2) == 3


def test_divide() -> None:
    assert divide(6, 2) == 3.0
