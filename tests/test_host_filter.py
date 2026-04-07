"""Tests for HostFilterMiddleware rejection path.

GAP-H2: All existing server tests use TEST_ALLOWED_HOSTS which includes
'testserver', so the middleware never rejects a request.  These tests
exercise the rejection path directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from keisei.db import init_db, write_training_state
from keisei.server.app import ALLOWED_HOSTS, create_app

pytestmark = pytest.mark.integration


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    init_db(path)
    write_training_state(path, {
        "config_json": "{}",
        "display_name": "TestBot",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-04-01T00:00:00Z",
    })
    return path


class TestHostFilterMiddleware:
    """Test that HostFilterMiddleware rejects unauthorized hosts."""

    @pytest.mark.asyncio
    async def test_blocked_host_returns_403(self, db_path: str) -> None:
        """Request with a host NOT in allowed set → 403 Forbidden."""
        # Use the real ALLOWED_HOSTS (no TEST_ALLOWED_HOSTS), and send
        # from a host that isn't in the allowlist.
        app = create_app(db_path, allowed_hosts=ALLOWED_HOSTS)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://evil.example.com",
        ) as client:
            resp = await client.get("/healthz")
        assert resp.status_code == 403
        assert resp.text == "Forbidden"

    @pytest.mark.asyncio
    async def test_allowed_host_passes(self, db_path: str) -> None:
        """Request with 'localhost' (in allowed set) → 200."""
        app = create_app(db_path, allowed_hosts=ALLOWED_HOSTS)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost",
        ) as client:
            resp = await client.get("/healthz")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_port_stripped_from_host(self, db_path: str) -> None:
        """Host 'localhost:8741' → strips port → 'localhost' → allowed."""
        app = create_app(db_path, allowed_hosts=ALLOWED_HOSTS)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost:8741",
        ) as client:
            resp = await client.get("/healthz")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_empty_host_blocked(self, db_path: str) -> None:
        """Empty host header → blocked (empty string not in allowlist)."""
        app = create_app(db_path, allowed_hosts=ALLOWED_HOSTS)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost",
            headers={"host": ""},
        ) as client:
            resp = await client.get("/healthz")
        assert resp.status_code == 403

    def test_websocket_blocked_host_closed(self, db_path: str) -> None:
        """WebSocket with blocked host → close code 1008."""
        # Use a custom allowed_hosts that does NOT include 'testserver'
        # (which is what TestClient sends by default).
        restricted_hosts = frozenset({"only-this-host.example.com"})
        app = create_app(db_path, allowed_hosts=restricted_hosts)
        client = TestClient(app)
        with pytest.raises(Exception):
            # TestClient sends Host: testserver, which isn't in restricted_hosts.
            # The server should close the WebSocket with code 1008.
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()  # Should not reach here

    def test_websocket_allowed_host_connects(self, db_path: str) -> None:
        """WebSocket with allowed host → accepts connection and sends init."""
        # Include 'testserver' in the allowlist (what TestClient sends)
        hosts_with_testserver = ALLOWED_HOSTS | frozenset({"testserver"})
        app = create_app(db_path, allowed_hosts=hosts_with_testserver)
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "init"

    @pytest.mark.asyncio
    async def test_custom_allowed_hosts_override(self, db_path: str) -> None:
        """create_app(allowed_hosts=...) overrides the default set."""
        custom_hosts = frozenset({"my-custom-host.internal"})
        app = create_app(db_path, allowed_hosts=custom_hosts)
        transport = ASGITransport(app=app)

        # Custom host → allowed
        async with AsyncClient(
            transport=transport,
            base_url="http://my-custom-host.internal",
        ) as client:
            resp = await client.get("/healthz")
        assert resp.status_code == 200

        # Default host → blocked
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost",
        ) as client:
            resp = await client.get("/healthz")
        assert resp.status_code == 403
