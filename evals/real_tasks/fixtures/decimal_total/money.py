def total_cents(prices: list[str]) -> int:
    return int(sum(float(price) for price in prices) * 100)
