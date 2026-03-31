import json

import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient

from keisei.server.app import create_app
from keisei.db import init_db, write_training_state, write_metrics


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


@pytest.mark.asyncio
async def test_healthz_ok(db_path: str) -> None:
    app = create_app(db_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db_accessible"] is True


@pytest.mark.asyncio
async def test_healthz_db_missing() -> None:
    app = create_app("/tmp/nonexistent-keisei-test.db")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["db_accessible"] is False


def test_ws_sends_init_on_connect(db_path: str) -> None:
    write_metrics(db_path, {"epoch": 0, "step": 100, "policy_loss": 1.5})
    app = create_app(db_path)
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "init"
        assert "games" in msg
        assert "metrics" in msg
        assert "training_state" in msg
        assert msg["training_state"]["display_name"] == "TestBot"
