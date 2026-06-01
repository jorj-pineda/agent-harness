"""Static demo panel mount (slice 9b).

The panel is served from the repo `ui/` directory via a catch-all StaticFiles
mount at `/`. These tests assert the mount is additive: API routes still win,
and the panel + assets are reachable.
"""

from __future__ import annotations

from .conftest import Harness, make_response


def test_root_serves_panel_html(harness: Harness) -> None:
    resp = harness.client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "agent-harness" in resp.text
    assert "/app.js" in resp.text


def test_static_assets_served(harness: Harness) -> None:
    js = harness.client.get("/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert "/sessions" in js.text  # the client posts here

    css = harness.client.get("/style.css")
    assert css.status_code == 200
    assert "css" in css.headers["content-type"]


def test_mount_does_not_shadow_api_routes(harness: Harness) -> None:
    """The catch-all UI mount must not intercept POST /sessions or /chat."""
    create = harness.client.post("/sessions", json={"user_id": "u-ui"})
    assert create.status_code == 200, create.text
    session_id = create.json()["session_id"]

    harness.provider.script(make_response(content="ok"))
    chat = harness.client.post(
        "/chat",
        json={"user_id": "u-ui", "session_id": session_id, "message": "hi"},
    )
    assert chat.status_code == 200, chat.text
    assert chat.json()["answer"] == "ok"


def test_openapi_still_reachable(harness: Harness) -> None:
    """FastAPI's own routes resolve ahead of the `/` mount."""
    resp = harness.client.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.json()["info"]["title"] == "agent-harness"
