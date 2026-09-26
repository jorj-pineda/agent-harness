from money import total_cents


def test_decimal_rounding() -> None:
    assert total_cents(["0.10", "0.20"]) == 30
    assert total_cents(["1.005"]) == 101
    assert total_cents(["-1.005"]) == -101
    assert total_cents([]) == 0
