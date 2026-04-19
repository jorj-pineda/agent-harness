---
title: Account Access and Recovery
category: account
---

# Account Access and Recovery

## Password reset

Customers request a password reset from the login page or by asking support. The reset email is sent to the address on file and expires in 30 minutes. Agents cannot view or set passwords directly — they can only trigger a reset email. If a customer reports never receiving the email, the first step is to confirm the email on file matches what they expect; mistyped addresses at signup are the most common cause.

## Locked accounts

An account locks automatically after 10 failed login attempts in 15 minutes. Locks clear after 30 minutes or can be cleared immediately by an agent after identity verification (order history + last-4 of payment method). Repeated lock events within a 24-hour window are flagged for security review and must be escalated.

## Two-factor authentication

Two-factor authentication is optional on standard accounts and mandatory on enterprise accounts. Supported methods are TOTP (authenticator app) and SMS. Agents cannot disable 2FA on a customer's behalf under any circumstance; customers who have lost access to their 2FA method must go through the account recovery flow, which requires government ID verification.

## Identity verification

Before making any account-level change (email change, 2FA reset, payment method removal), agents must verify identity using two of the following: the last 4 digits of a payment method on file, the order ID of a recent order, and the email address on file. Enterprise accounts additionally require the account admin's employee ID.
