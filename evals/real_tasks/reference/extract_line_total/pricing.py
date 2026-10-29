def line_total(item):
    return item["unit_cents"] * item["quantity"]


def subtotal(items):
    return sum(line_total(item) for item in items)
