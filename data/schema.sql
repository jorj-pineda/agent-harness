-- Mock e-commerce customer-support schema.
-- The SQL tool (step 5) queries this via a read-only connection.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
    id         INTEGER PRIMARY KEY,
    email      TEXT NOT NULL UNIQUE,
    name       TEXT NOT NULL,
    tier       TEXT NOT NULL CHECK (tier IN ('standard', 'premium', 'enterprise')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id           INTEGER PRIMARY KEY,
    sku          TEXT NOT NULL UNIQUE,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,
    price_cents  INTEGER NOT NULL CHECK (price_cents >= 0),
    in_stock     INTEGER NOT NULL DEFAULT 1 CHECK (in_stock IN (0, 1))
);

CREATE TABLE IF NOT EXISTS orders (
    id            INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(id),
    status        TEXT NOT NULL CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled', 'refunded')),
    total_cents   INTEGER NOT NULL CHECK (total_cents >= 0),
    placed_at     TEXT NOT NULL,
    shipped_at    TEXT,
    delivered_at  TEXT
);

CREATE TABLE IF NOT EXISTS order_items (
    id               INTEGER PRIMARY KEY,
    order_id         INTEGER NOT NULL REFERENCES orders(id),
    product_id       INTEGER NOT NULL REFERENCES products(id),
    quantity         INTEGER NOT NULL CHECK (quantity > 0),
    unit_price_cents INTEGER NOT NULL CHECK (unit_price_cents >= 0)
);

CREATE TABLE IF NOT EXISTS tickets (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_id    INTEGER REFERENCES orders(id),
    subject     TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('open', 'pending_customer', 'resolved', 'escalated')),
    priority    TEXT NOT NULL CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    created_at  TEXT NOT NULL,
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_customer  ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status   ON tickets(status);
