from __future__ import annotations

import logging
from pathlib import Path

import pytest

from data.seed import seed
from data.sqlite import init_schema, open_rw
from tools import ToolError, ToolRegistry
from tools.sql import (
    MAX_ROWS,
    build_sql_tools,
    register_sql_tools,
)


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "support.db"
    conn = open_rw(db_path)
    try:
        init_schema(conn)
        seed(conn)
    finally:
        conn.close()
    return db_path


@pytest.fixture
def registry(seeded_db: Path) -> ToolRegistry:
    reg = ToolRegistry()
    register_sql_tools(reg, db_path=seeded_db)
    return reg


async def test_lookup_customer_by_email_hit(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "lookup_customer_by_email",
        {"email": "alice.chen@example.com"},
    )
    assert result == {
        "id": 1,
        "email": "alice.chen@example.com",
        "name": "Alice Chen",
        "tier": "standard",
        "created_at": "2025-11-04T14:22:10Z",
    }


async def test_lookup_customer_by_email_miss(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "lookup_customer_by_email",
        {"email": "nobody@example.com"},
    )
    assert result is None


async def test_lookup_order_includes_line_items_and_customer(registry: ToolRegistry) -> None:
    result = await registry.invoke("lookup_order", {"order_id": 1001})
    assert result is not None
    assert result["id"] == 1001
    assert result["customer_email"] == "alice.chen@example.com"
    assert result["customer_name"] == "Alice Chen"
    assert result["status"] == "delivered"
    assert len(result["items"]) == 2
    skus = {item["sku"] for item in result["items"]}
    assert skus == {"ECHO-01", "TEE-BLK"}


async def test_lookup_order_miss(registry: ToolRegistry) -> None:
    assert await registry.invoke("lookup_order", {"order_id": 999999}) is None


async def test_list_customer_orders_newest_first(registry: ToolRegistry) -> None:
    # Alice (id=1) has orders 1001 (2026-01-12) and 1002 (2026-04-10).
    result = await registry.invoke("list_customer_orders", {"customer_id": 1})
    assert [o["id"] for o in result] == [1002, 1001]


async def test_list_customer_orders_status_filter(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "list_customer_orders",
        {"customer_id": 1, "status": "delivered"},
    )
    assert [o["id"] for o in result] == [1001]


async def test_list_customer_orders_respects_max_results(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "list_customer_orders",
        {"customer_id": 1, "max_results": 1},
    )
    assert len(result) == 1
    assert result[0]["id"] == 1002


async def test_list_customer_orders_missing_customer_is_empty(registry: ToolRegistry) -> None:
    assert await registry.invoke("list_customer_orders", {"customer_id": 99999}) == []


async def test_list_customer_tickets_newest_first(registry: ToolRegistry) -> None:
    # Bob (id=2) has tickets 1 (2026-03-05) and 9 (2026-02-28).
    result = await registry.invoke("list_customer_tickets", {"customer_id": 2})
    assert [t["id"] for t in result] == [1, 9]


async def test_list_customer_tickets_status_filter(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "list_customer_tickets",
        {"customer_id": 7, "status": "escalated"},
    )
    assert [t["id"] for t in result] == [6]


async def test_lookup_product_by_sku_hit(registry: ToolRegistry) -> None:
    result = await registry.invoke("lookup_product", {"sku": "ECHO-01"})
    assert result is not None
    assert result["name"] == "Echo Earbuds Pro"
    assert result["price_cents"] == 17999
    assert result["in_stock"] == 1


async def test_lookup_product_miss(registry: ToolRegistry) -> None:
    assert await registry.invoke("lookup_product", {"sku": "DOES-NOT-EXIST"}) is None


def test_register_sql_tools_exposes_every_tool(seeded_db: Path) -> None:
    reg = ToolRegistry()
    register_sql_tools(reg, db_path=seeded_db)
    assert sorted(reg.names()) == [
        "list_customer_orders",
        "list_customer_tickets",
        "lookup_customer_by_email",
        "lookup_order",
        "lookup_product",
    ]


def test_every_tool_produces_a_valid_provider_spec(seeded_db: Path) -> None:
    # Round-trips through to_spec() to confirm input_model is Pydantic-valid
    # and will serialize into the JSON-schema shape the providers expect.
    for t in build_sql_tools(seeded_db):
        spec = t.to_spec()
        assert spec.name == t.name
        assert spec.description
        assert spec.parameters_schema["type"] == "object"


async def test_max_results_over_cap_is_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError, match="Invalid arguments"):
        await registry.invoke(
            "list_customer_orders",
            {"customer_id": 1, "max_results": MAX_ROWS + 1},
        )


async def test_invalid_status_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError, match="Invalid arguments"):
        await registry.invoke(
            "list_customer_orders",
            {"customer_id": 1, "status": "not-a-real-status"},
        )


async def test_negative_customer_id_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError, match="Invalid arguments"):
        await registry.invoke("list_customer_orders", {"customer_id": -1})


async def test_every_tool_call_is_logged(
    registry: ToolRegistry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="tools.sql")
    await registry.invoke("lookup_product", {"sku": "ECHO-01"})
    await registry.invoke("lookup_customer_by_email", {"email": "x@example.com"})
    await registry.invoke("list_customer_orders", {"customer_id": 1, "max_results": 5})
    messages = [r.getMessage() for r in caplog.records if r.name == "tools.sql"]
    assert any("sql_tool=lookup_product" in m for m in messages)
    assert any("sql_tool=lookup_customer_by_email" in m for m in messages)
    assert any("sql_tool=list_customer_orders" in m for m in messages)


async def test_missing_db_surfaces_as_tool_error(tmp_path: Path) -> None:
    reg = ToolRegistry()
    register_sql_tools(reg, db_path=tmp_path / "nowhere.db")
    with pytest.raises(ToolError, match="raised"):
        await reg.invoke("lookup_product", {"sku": "ECHO-01"})


async def test_read_only_connection_blocks_writes(seeded_db: Path) -> None:
    """Sanity check: the read-only open flag really blocks mutations.

    The SQL tools don't accept arbitrary SQL, but this test pins down the
    guarantee we rely on at the connection layer — if it ever regresses,
    the whole rule-#4 story breaks silently.
    """
    import sqlite3

    from data.sqlite import open_ro

    with open_ro(seeded_db) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("UPDATE customers SET name = 'hax' WHERE id = 1")
