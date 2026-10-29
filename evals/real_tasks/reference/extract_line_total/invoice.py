import pricing


def invoice_lines(items):
    return [pricing.line_total(item) for item in items]
