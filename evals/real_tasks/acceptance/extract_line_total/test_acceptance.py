import invoice
import pricing
import pytest


def test_integer_arithmetic_empty_order_iterables_and_preservation():
    items = [
        {"unit_cents": 125, "quantity": 2},
        {"unit_cents": 17, "quantity": -3},
        {"unit_cents": 99, "quantity": 0},
    ]
    before = [item.copy() for item in items]
    assert [pricing.line_total(item) for item in items] == [250, -51, 0]
    assert pricing.subtotal(iter(items)) == 199
    assert invoice.invoice_lines(iter(items)) == [250, -51, 0]
    assert pricing.subtotal([]) == 0 and invoice.invoice_lines([]) == []
    assert items == before
    large = {"unit_cents": 10**30 + 1, "quantity": 3}
    expected = (10**30 + 1) * 3
    assert type(pricing.line_total(large)) is int and pricing.line_total(large) == expected
    assert pricing.subtotal([large]) == expected
    assert invoice.invoice_lines([large]) == [expected]


def test_both_modules_use_current_helper_at_call_time(monkeypatch):
    calls = []

    def replacement(item):
        calls.append(item)
        return item["custom"]

    monkeypatch.setattr(pricing, "line_total", replacement)
    items = [{"custom": 8}, {"custom": -2}]
    assert pricing.subtotal(iter(items)) == 6
    assert invoice.invoice_lines(iter(items)) == [8, -2]
    assert all(a is b for a, b in zip(calls, items + items, strict=True))


@pytest.mark.parametrize(
    "item,error",
    [
        ({}, KeyError),
        ({"unit_cents": 1}, KeyError),
        ({"unit_cents": None, "quantity": 1}, TypeError),
    ],
)
def test_existing_errors_propagate(item, error):
    for function in (
        pricing.line_total,
        lambda value: pricing.subtotal([value]),
        lambda value: invoice.invoice_lines([value]),
    ):
        with pytest.raises(error):
            function(item)
