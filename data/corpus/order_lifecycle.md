---
title: Order Lifecycle and Status
category: orders
---

# Order Lifecycle and Status

## Status values

Orders move through five statuses: `pending` (payment authorized, not yet shipped), `shipped` (package handed to carrier, tracking number issued), `delivered` (carrier confirmed delivery), `cancelled` (cancelled before shipment), and `refunded` (returned and refund issued). Only `pending` orders can be cancelled from the account portal; post-shipment cancellations require an agent and are handled as refusals or returns.

## Cancellation window

Customers can cancel a `pending` order from the account portal for up to 60 minutes after placement. After that, cancellation requires an agent, and the order can still be cancelled only while status remains `pending`. Once the order transitions to `shipped`, it must be handled as a return once delivered or refused at the door.

## Modifications after placement

We do not allow editing an order after placement. If a customer wants to change items, quantities, or the shipping address, the correct flow is: cancel the original (if still `pending`) and place a new order. Address changes on already-shipped orders require intercepting the package with the carrier, which is not always possible and is not guaranteed.

## Delivered but not received

When an order shows `delivered` but the customer reports they did not receive it, first confirm the shipping address on file, then check whether the carrier photo or signature is available. If neither confirms receipt by the customer, file a carrier claim and offer the customer a replacement. Do not issue a cash refund on a first report of non-delivery; a replacement keeps the claim investigable.
