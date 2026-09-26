from decimal import ROUND_HALF_UP, Decimal


def total_cents(prices: list[str]) -> int:
    total = sum((Decimal(price) for price in prices), Decimal("0"))
    return int((total * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
