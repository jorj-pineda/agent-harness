from __future__ import annotations

from pathlib import Path

from data.seed import (
    CUSTOMERS,
    ORDER_ITEMS,
    ORDERS,
    PRODUCTS,
    TICKETS,
    seed,
)
from data.sqlite import init_schema, open_rw


def _seeded_db(tmp_path: Path):
    conn = open_rw(tmp_path / "seeded.db")
    init_schema(conn)
    seed(conn)
    return conn


def test_seed_inserts_expected_row_counts(tmp_path: Path) -> None:
    conn = _seeded_db(tmp_path)
    try:
        counts = {
            "customers": conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0],
            "products": conn.execute("SELECT COUNT(*) FROM products").fetchone()[0],
            "orders": conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
            "order_items": conn.execute("SELECT COUNT(*) FROM order_items").fetchone()[0],
            "tickets": conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0],
        }
        assert counts == {
            "customers": len(CUSTOMERS),
            "products": len(PRODUCTS),
            "orders": len(ORDERS),
            "order_items": len(ORDER_ITEMS),
            "tickets": len(TICKETS),
        }
    finally:
        conn.close()


def test_seed_is_idempotent(tmp_path: Path) -> None:
    conn = _seeded_db(tmp_path)
    try:
        seed(conn)  # re-run
        seed(conn)  # and again
        assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == len(CUSTOMERS)
        assert conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == len(TICKETS)
    finally:
        conn.close()


def test_seed_covers_every_customer_tier(tmp_path: Path) -> None:
    conn = _seeded_db(tmp_path)
    try:
        tiers = {r[0] for r in conn.execute("SELECT DISTINCT tier FROM customers")}
        assert tiers == {"standard", "premium", "enterprise"}
    finally:
        conn.close()


def test_seed_covers_every_order_status(tmp_path: Path) -> None:
    conn = _seeded_db(tmp_path)
    try:
        statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM orders")}
        assert statuses == {"pending", "shipped", "delivered", "cancelled", "refunded"}
    finally:
        conn.close()


def test_seed_covers_every_ticket_status_and_priority(tmp_path: Path) -> None:
    conn = _seeded_db(tmp_path)
    try:
        statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM tickets")}
        priorities = {r[0] for r in conn.execute("SELECT DISTINCT priority FROM tickets")}
        assert statuses == {"open", "pending_customer", "resolved", "escalated"}
        # Not every priority needs to appear, but we want at least 3 of 4 for demo variety.
        assert len(priorities) >= 3
    finally:
        conn.close()


def test_seed_has_out_of_stock_product(tmp_path: Path) -> None:
    """Grounding/RAG scenarios need an out-of-stock product to exercise the
    "can't fulfill" escalation path, so the seed must always include one."""
    conn = _seeded_db(tmp_path)
    try:
        (n,) = conn.execute("SELECT COUNT(*) FROM products WHERE in_stock = 0").fetchone()
        assert n >= 1
    finally:
        conn.close()


def test_seed_respects_foreign_keys(tmp_path: Path) -> None:
    conn = _seeded_db(tmp_path)
    try:
        # Every order_item references an existing order and product.
        orphaned_items = conn.execute(
            "SELECT COUNT(*) FROM order_items oi "
            "LEFT JOIN orders o ON o.id = oi.order_id "
            "LEFT JOIN products p ON p.id = oi.product_id "
            "WHERE o.id IS NULL OR p.id IS NULL"
        ).fetchone()[0]
        assert orphaned_items == 0

        # Every order references an existing customer.
        orphaned_orders = conn.execute(
            "SELECT COUNT(*) FROM orders o "
            "LEFT JOIN customers c ON c.id = o.customer_id "
            "WHERE c.id IS NULL"
        ).fetchone()[0]
        assert orphaned_orders == 0

        # Every ticket.order_id either is NULL or references an existing order.
        orphaned_tickets = conn.execute(
            "SELECT COUNT(*) FROM tickets t "
            "LEFT JOIN orders o ON o.id = t.order_id "
            "WHERE t.order_id IS NOT NULL AND o.id IS NULL"
        ).fetchone()[0]
        assert orphaned_tickets == 0
    finally:
        conn.close()


def test_seed_all_customers_have_email(tmp_path: Path) -> None:
    conn = _seeded_db(tmp_path)
    try:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM customers WHERE email IS NULL OR email = ''"
        ).fetchone()
        assert n == 0
    finally:
        conn.close()
