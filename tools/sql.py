"""Read-only SQL tools for the customer-support harness.

Each tool is a narrowly-typed, Pydantic-validated query against the seeded
SQLite support DB. No arbitrary SQL is ever accepted — the agent picks a
tool by name, supplies parameters, and gets back a structured row (or null).

CLAUDE.md rule #4 compliance:
  * Read-only connection (`data.sqlite.open_ro` → sqlite3 URI `mode=ro`,
    so writes are blocked at the OS file layer, not just the library).
  * Every query is parameterized; user input never concatenates into SQL.
  * List tools cap `max_results` at the Pydantic layer and bind it to
    `LIMIT ?` server-side.
  * Every invocation is logged on the `tools.sql` logger.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from data.sqlite import open_ro

from .base import Tool
from .registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_ROWS = 50
DEFAULT_ROWS = 10

OrderStatus = Literal["pending", "shipped", "delivered", "cancelled", "refunded"]
TicketStatus = Literal["open", "pending_customer", "resolved", "escalated"]


class LookupCustomerByEmailInput(BaseModel):
    email: str = Field(..., min_length=3, description="Email address (exact match)")


class LookupOrderInput(BaseModel):
    order_id: int = Field(..., ge=1, description="Numeric order ID")


class ListCustomerOrdersInput(BaseModel):
    customer_id: int = Field(..., ge=1)
    status: OrderStatus | None = Field(None, description="Optional status filter")
    max_results: int = Field(DEFAULT_ROWS, ge=1, le=MAX_ROWS)


class ListCustomerTicketsInput(BaseModel):
    customer_id: int = Field(..., ge=1)
    status: TicketStatus | None = Field(None, description="Optional status filter")
    max_results: int = Field(DEFAULT_ROWS, ge=1, le=MAX_ROWS)


class LookupProductInput(BaseModel):
    sku: str = Field(..., min_length=1, description="Product SKU (exact match)")


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _log_call(tool_name: str, **params: Any) -> None:
    log.info("sql_tool=%s args=%s", tool_name, params)


def build_sql_tools(db_path: Path | str) -> list[Tool]:
    """Build SQL Tool instances bound to a specific DB file.

    Returned tools are plain objects — the caller decides which registry to
    register them on. This keeps tests free of module-level state and lets
    the harness swap DB paths per deployment without re-importing tools.
    """
    path = Path(db_path)

    def lookup_customer_by_email(args: LookupCustomerByEmailInput) -> dict[str, Any] | None:
        """Look up a customer by exact email. Returns the customer record or null."""
        _log_call("lookup_customer_by_email", email=args.email)
        with contextlib.closing(open_ro(path)) as conn:
            row = conn.execute(
                "SELECT id, email, name, tier, created_at FROM customers WHERE email = ?",
                (args.email,),
            ).fetchone()
        return _row(row)

    def lookup_order(args: LookupOrderInput) -> dict[str, Any] | None:
        """Fetch an order with line items and customer contact. Null if not found."""
        _log_call("lookup_order", order_id=args.order_id)
        with contextlib.closing(open_ro(path)) as conn:
            order = conn.execute(
                "SELECT o.id, o.customer_id, c.email AS customer_email, "
                "c.name AS customer_name, o.status, o.total_cents, "
                "o.placed_at, o.shipped_at, o.delivered_at "
                "FROM orders o JOIN customers c ON c.id = o.customer_id "
                "WHERE o.id = ?",
                (args.order_id,),
            ).fetchone()
            if order is None:
                return None
            items = conn.execute(
                "SELECT oi.product_id, p.sku, p.name AS product_name, "
                "oi.quantity, oi.unit_price_cents "
                "FROM order_items oi JOIN products p ON p.id = oi.product_id "
                "WHERE oi.order_id = ? ORDER BY oi.id",
                (args.order_id,),
            ).fetchall()
        result = dict(order)
        result["items"] = [dict(i) for i in items]
        return result

    def list_customer_orders(args: ListCustomerOrdersInput) -> list[dict[str, Any]]:
        """List a customer's orders, newest first; optional status filter."""
        _log_call(
            "list_customer_orders",
            customer_id=args.customer_id,
            status=args.status,
            max_results=args.max_results,
        )
        with contextlib.closing(open_ro(path)) as conn:
            if args.status is None:
                rows = conn.execute(
                    "SELECT id, status, total_cents, placed_at, shipped_at, delivered_at "
                    "FROM orders WHERE customer_id = ? "
                    "ORDER BY placed_at DESC LIMIT ?",
                    (args.customer_id, args.max_results),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, status, total_cents, placed_at, shipped_at, delivered_at "
                    "FROM orders WHERE customer_id = ? AND status = ? "
                    "ORDER BY placed_at DESC LIMIT ?",
                    (args.customer_id, args.status, args.max_results),
                ).fetchall()
        return [dict(r) for r in rows]

    def list_customer_tickets(args: ListCustomerTicketsInput) -> list[dict[str, Any]]:
        """List a customer's tickets, newest first; optional status filter."""
        _log_call(
            "list_customer_tickets",
            customer_id=args.customer_id,
            status=args.status,
            max_results=args.max_results,
        )
        with contextlib.closing(open_ro(path)) as conn:
            if args.status is None:
                rows = conn.execute(
                    "SELECT id, order_id, subject, status, priority, "
                    "created_at, resolved_at FROM tickets WHERE customer_id = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (args.customer_id, args.max_results),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, order_id, subject, status, priority, "
                    "created_at, resolved_at FROM tickets "
                    "WHERE customer_id = ? AND status = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (args.customer_id, args.status, args.max_results),
                ).fetchall()
        return [dict(r) for r in rows]

    def lookup_product(args: LookupProductInput) -> dict[str, Any] | None:
        """Look up a product by SKU. Returns the product record or null."""
        _log_call("lookup_product", sku=args.sku)
        with contextlib.closing(open_ro(path)) as conn:
            row = conn.execute(
                "SELECT id, sku, name, category, price_cents, in_stock FROM products WHERE sku = ?",
                (args.sku,),
            ).fetchone()
        return _row(row)

    return [
        Tool(
            name="lookup_customer_by_email",
            description=(
                "Look up a customer record by exact email match. "
                "Returns the customer fields or null if no match."
            ),
            input_model=LookupCustomerByEmailInput,
            fn=lookup_customer_by_email,
        ),
        Tool(
            name="lookup_order",
            description=(
                "Fetch a single order by its numeric ID, including the line items "
                "and the customer's name/email. Returns null if no such order."
            ),
            input_model=LookupOrderInput,
            fn=lookup_order,
        ),
        Tool(
            name="list_customer_orders",
            description=(
                "List a customer's orders, newest first. Optional status filter "
                "(pending/shipped/delivered/cancelled/refunded). Capped at 50 rows."
            ),
            input_model=ListCustomerOrdersInput,
            fn=list_customer_orders,
        ),
        Tool(
            name="list_customer_tickets",
            description=(
                "List a customer's support tickets, newest first. Optional status filter "
                "(open/pending_customer/resolved/escalated). Capped at 50 rows."
            ),
            input_model=ListCustomerTicketsInput,
            fn=list_customer_tickets,
        ),
        Tool(
            name="lookup_product",
            description=(
                "Look up a product by its SKU (exact match). "
                "Returns product fields including price and stock, or null."
            ),
            input_model=LookupProductInput,
            fn=lookup_product,
        ),
    ]


def register_sql_tools(registry: ToolRegistry, *, db_path: Path | str) -> None:
    """Register every SQL tool on the given registry."""
    for t in build_sql_tools(db_path):
        registry.register(t)
