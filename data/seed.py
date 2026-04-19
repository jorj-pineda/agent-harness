"""Populate the mock support database with a curated, deterministic dataset.

Data is hardcoded (no RNG) so evaluation scenarios can assert against
specific rows without flakiness. Re-running wipes the five tables first,
so `python -m data.seed` is always idempotent.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from .sqlite import init_schema, open_rw

# fmt: off
# (id, email, name, tier, created_at)
CUSTOMERS: tuple[tuple[int, str, str, str, str], ...] = (
    (1, "alice.chen@example.com",   "Alice Chen",   "standard",   "2025-11-04T14:22:10Z"),
    (2, "bob.kowalski@example.com", "Bob Kowalski", "premium",    "2025-03-18T09:05:00Z"),
    (3, "carol.dvorak@example.com", "Carol Dvorak", "enterprise", "2024-07-22T16:40:55Z"),
    (4, "david.patel@example.com",  "David Patel",  "standard",   "2025-08-02T11:17:30Z"),
    (5, "eve.nakamura@example.com", "Eve Nakamura", "premium",    "2024-12-15T20:11:45Z"),
    (6, "frank.oduya@example.com",  "Frank Oduya",  "standard",   "2026-01-09T08:02:13Z"),
    (7, "grace.hopper@example.com", "Grace Hopper", "enterprise", "2023-05-30T12:00:00Z"),
    (8, "henry.tanaka@example.com", "Henry Tanaka", "standard",   "2026-04-10T18:55:21Z"),
)

# (id, sku, name, category, price_cents, in_stock)
PRODUCTS: tuple[tuple[int, str, str, str, int, int], ...] = (
    (1,  "ECHO-01",  "Echo Earbuds Pro",    "electronics", 17999, 1),
    (2,  "TAB-10",   'TabletView 10"',      "electronics", 34900, 1),
    (3,  "KBD-X",    "MechaKey X",          "electronics", 12950, 1),
    (4,  "MON-27",   "UltraMonitor 27",     "electronics", 44900, 0),
    (5,  "TEE-BLK",  "Classic Tee (Black)", "apparel",      2499, 1),
    (6,  "HOOD-GRY", "Pullover Hoodie",     "apparel",      5499, 1),
    (7,  "JKT-RED",  "Windbreaker Red",     "apparel",      8999, 1),
    (8,  "CAP-NVY",  "Logo Cap Navy",       "apparel",      1999, 1),
    (9,  "MUG-CER",  "Ceramic Mug",         "home",         1499, 1),
    (10, "LMP-DSK",  "Desk Lamp LED",       "home",         3999, 1),
    (11, "BAG-TOT",  "Cotton Tote",         "home",         1299, 0),
    (12, "BOT-STL",  "Steel Bottle",        "home",         2299, 1),
)

# (id, customer_id, status, total_cents, placed_at, shipped_at, delivered_at)
ORDERS: tuple[tuple[int, int, str, int, str, str | None, str | None], ...] = (
    (1001, 1, "delivered", 20498, "2026-01-12T10:15:00Z", "2026-01-13T09:00:00Z", "2026-01-15T14:30:00Z"),
    (1002, 1, "shipped",    2499, "2026-04-10T08:30:00Z", "2026-04-11T07:45:00Z", None),
    (1003, 2, "delivered", 34900, "2026-02-02T19:40:00Z", "2026-02-03T11:00:00Z", "2026-02-05T16:12:00Z"),
    (1004, 2, "refunded",  12950, "2026-03-01T12:00:00Z", "2026-03-02T10:00:00Z", "2026-03-04T15:00:00Z"),
    (1005, 3, "delivered", 44900, "2025-12-20T09:22:00Z", "2025-12-21T08:00:00Z", "2025-12-23T17:45:00Z"),
    (1006, 3, "pending",    5998, "2026-04-17T22:11:00Z", None, None),
    (1007, 4, "cancelled",  8999, "2026-03-28T14:05:00Z", None, None),
    (1008, 4, "delivered",  4498, "2026-02-14T11:30:00Z", "2026-02-15T10:00:00Z", "2026-02-17T13:20:00Z"),
    (1009, 5, "shipped",   22798, "2026-04-14T16:50:00Z", "2026-04-15T09:10:00Z", None),
    (1010, 5, "delivered", 17999, "2026-01-30T07:00:00Z", "2026-01-31T08:45:00Z", "2026-02-02T12:00:00Z"),
    (1011, 6, "delivered",  1499, "2026-03-15T13:25:00Z", "2026-03-16T09:30:00Z", "2026-03-18T11:00:00Z"),
    (1012, 6, "pending",    3998, "2026-04-16T10:00:00Z", None, None),
    (1013, 7, "delivered", 54899, "2025-10-05T08:15:00Z", "2025-10-06T07:30:00Z", "2025-10-08T14:40:00Z"),
    (1014, 7, "shipped",   12950, "2026-04-12T11:22:00Z", "2026-04-13T08:05:00Z", None),
    (1015, 8, "delivered",  4498, "2026-04-11T17:45:00Z", "2026-04-12T09:00:00Z", "2026-04-14T10:15:00Z"),
)

# (id, order_id, product_id, quantity, unit_price_cents)
ORDER_ITEMS: tuple[tuple[int, int, int, int, int], ...] = (
    (1,  1001, 1,  1, 17999),
    (2,  1001, 5,  1,  2499),
    (3,  1002, 5,  1,  2499),
    (4,  1003, 2,  1, 34900),
    (5,  1004, 3,  1, 12950),
    (6,  1005, 4,  1, 44900),
    (7,  1006, 9,  2,  1499),
    (8,  1006, 10, 1,  3000),  # promotional price noted in ticket 5
    (9,  1007, 7,  1,  8999),
    (10, 1008, 6,  1,  5499),
    (11, 1008, 8,  1,  1999),
    (12, 1009, 6,  1,  5499),
    (13, 1009, 12, 1,  2299),
    (14, 1009, 7,  1,  8999),
    (15, 1009, 8,  3,  2000),
    (16, 1010, 1,  1, 17999),
    (17, 1011, 9,  1,  1499),
    (18, 1012, 10, 1,  3999),
    (19, 1013, 2,  1, 34900),
    (20, 1013, 6,  1,  5499),
    (21, 1013, 3,  1, 12950),
    (22, 1013, 7,  1,  1550),
    (23, 1014, 3,  1, 12950),
    (24, 1015, 6,  1,  2499),
    (25, 1015, 8,  1,  1999),
)

# (id, customer_id, order_id, subject, status, priority, created_at, resolved_at)
TICKETS: tuple[tuple[int, int, int | None, str, str, str, str, str | None], ...] = (
    (1,  2, 1004, "Defective keyboard - keys stuck on arrival", "resolved",         "high",   "2026-03-05T09:14:00Z", "2026-03-07T16:20:00Z"),
    (2,  4, 1007, "Cancelled order - did not receive refund",   "escalated",        "urgent", "2026-04-01T11:30:00Z", None),
    (3,  1, 1002, "Where is my order? Still marked shipped",    "open",             "normal", "2026-04-17T19:22:00Z", None),
    (4,  5, None, "Request to upgrade account to enterprise",   "pending_customer", "low",    "2026-04-14T08:05:00Z", None),
    (5,  3, 1006, "Missing promo discount on Desk Lamp",        "open",             "normal", "2026-04-18T07:40:00Z", None),
    (6,  7, 1013, "Warranty claim - monitor flickering",        "escalated",        "high",   "2026-03-22T14:00:00Z", None),
    (7,  6, 1011, "Mug arrived cracked",                        "resolved",         "low",    "2026-03-19T10:10:00Z", "2026-03-20T12:45:00Z"),
    (8,  8, None, "How do I track my account's open orders?",   "resolved",         "low",    "2026-04-13T15:33:00Z", "2026-04-13T16:00:00Z"),
    (9,  2, None, "Invoice export for Feb tax filing",          "resolved",         "normal", "2026-02-28T11:11:00Z", "2026-03-01T09:22:00Z"),
    (10, 3, 1005, "Bulk pricing for repeat monitor order",      "pending_customer", "normal", "2026-04-09T13:18:00Z", None),
)
# fmt: on

_TABLES_IN_DEPENDENCY_ORDER = ("tickets", "order_items", "orders", "products", "customers")


def seed(conn: sqlite3.Connection) -> None:
    """Wipe and repopulate every table from the constants above."""
    for table in _TABLES_IN_DEPENDENCY_ORDER:
        conn.execute(f"DELETE FROM {table}")  # noqa: S608  (table names are internal constants)

    conn.executemany(
        "INSERT INTO customers (id, email, name, tier, created_at) VALUES (?, ?, ?, ?, ?)",
        CUSTOMERS,
    )
    conn.executemany(
        "INSERT INTO products (id, sku, name, category, price_cents, in_stock) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        PRODUCTS,
    )
    conn.executemany(
        "INSERT INTO orders "
        "(id, customer_id, status, total_cents, placed_at, shipped_at, delivered_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ORDERS,
    )
    conn.executemany(
        "INSERT INTO order_items (id, order_id, product_id, quantity, unit_price_cents) "
        "VALUES (?, ?, ?, ?, ?)",
        ORDER_ITEMS,
    )
    conn.executemany(
        "INSERT INTO tickets "
        "(id, customer_id, order_id, subject, status, priority, created_at, resolved_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        TICKETS,
    )
    conn.commit()


def main() -> None:
    db_path = Path(os.getenv("SQLITE_DB_PATH", "data/support.db"))
    conn = open_rw(db_path)
    try:
        init_schema(conn)
        seed(conn)
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
            for table in _TABLES_IN_DEPENDENCY_ORDER
        }
    finally:
        conn.close()
    print(f"Seeded {db_path}:")
    for table, n in counts.items():
        print(f"  {table:>12}: {n}")


if __name__ == "__main__":
    main()
