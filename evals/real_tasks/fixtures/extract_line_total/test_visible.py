import pricing
from invoice import invoice_lines


def test_line_helper_and_existing_callers():
    items = [{"unit_cents": 125, "quantity": 2}]
    assert pricing.line_total(items[0]) == 250
    assert pricing.subtotal(items) == 250
    assert invoice_lines(items) == [250]
