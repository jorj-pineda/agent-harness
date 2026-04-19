---
title: Payments and Refund Timing
category: payments
---

# Payments and Refund Timing

## Accepted payment methods

We accept Visa, Mastercard, American Express, Discover, Apple Pay, Google Pay, and ACH transfers for US customers. Enterprise accounts additionally support invoice billing with NET-30 terms. We do not accept cryptocurrency, personal checks, or money orders.

## Refund processing windows

Once a refund is approved, the platform-side processing happens within 1 business day. After that, the time to appear in the customer's account depends on the payment method: credit cards take 5–10 business days, ACH takes 3–5 business days, Apple Pay and Google Pay follow the underlying card's timing, and invoice-billed accounts see the credit on the next monthly statement.

## Partial refunds

Partial refunds are issued when only part of an order is returned or when a price-match adjustment is granted. The refund amount is calculated against the line-item `unit_price_cents` on the original order, not the current catalog price. Shipping fees are refunded only if the entire order is returned or if the return was caused by our error.

## Disputes and chargebacks

If a customer initiates a chargeback with their card issuer before contacting support, the case moves out of our normal refund workflow and into the payments team queue. Agents must not issue a manual refund on an order that already has an active chargeback — doing so results in a double refund that is difficult to reverse. Check the order's `chargeback_status` flag before processing any refund.
