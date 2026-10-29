def invoice_lines(items):
    return [item["unit_cents"] * item["quantity"] for item in items]
