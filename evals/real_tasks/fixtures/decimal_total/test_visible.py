from money import total_cents


def test_total() -> None:
    assert total_cents(["0.29"]) == 29
