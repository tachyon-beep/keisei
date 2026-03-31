# Plan 1: FastAPI Server

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FastAPI server that reads from SQLite and pushes game state + metrics to browser clients via WebSocket.

**Architecture:** Single-file FastAPI app with `lifespan` context manager, async SQLite reads via `asyncio.to_thread`, WebSocket lifecycle via `asyncio.TaskGroup`, high-water mark polling.

**Tech Stack:** FastAPI, uvicorn, asyncio, sqlite3

---

### Task 1: Server Package Scaffolding

**Files:**
- Create: `keisei/server/__init__.py`
- Create: `keisei/server/app.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: Create server package**

`keisei/server/__init__.py`:
```python
"""FastAPI spectator dashboard server."""
```

- [ ] **Step 2: Write failing test for healthz endpoint**

`tests/test_server.py`:
```python
import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from keisei.server.app import create_app
from keisei.db import init_db, write_training_state


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement app.py with healthz endpoint**

`keisei/server/app.py`:
```python
"""FastAPI spectator dashboard server."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from keisei.db import read_metrics_since, read_game_snapshots, read_training_state

logger = logging.getLogger(__name__)

MAX_METRICS_IN_INIT = 500
POLL_INTERVAL_S = 0.2
HEARTBEAT_STALE_S = 30
WS_SEND_TIMEOUT_S = 5.0
WS_PING_INTERVAL_S = 15.0


def _db_accessible(db_path: str) -> bool:
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("SELECT 1 FROM schema_version")
        conn.close()
        return True
    except Exception:
        return False


def _training_alive(db_path: str) -> bool:
    try:
        state = read_training_state(db_path)
        if state is None:
            return False
        hb = state.get("heartbeat_at", "")
        if not hb:
            return False
        hb_time = datetime.fromisoformat(hb.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - hb_time).total_seconds()
        return age < HEARTBEAT_STALE_S
    except Exception:
        return False


def create_app(db_path: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Server starting, db_path=%s", db_path)
        yield
        logger.info("Server shutting down")

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        accessible = await asyncio.to_thread(_db_accessible, db_path)
        alive = await asyncio.to_thread(_training_alive, db_path) if accessible else False
        return JSONResponse({
            "status": "ok",
            "db_accessible": accessible,
            "training_alive": alive,
        })

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(_poll_and_push(websocket, db_path))
                tg.create_task(_keepalive(websocket))
        except* WebSocketDisconnect:
            pass
        except* Exception as eg:
            for exc in eg.exceptions:
                if not isinstance(exc, WebSocketDisconnect):
                    logger.warning("WebSocket error: %s", exc)

    # Mount static files if the directory exists
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


async def _poll_and_push(ws: WebSocket, db_path: str) -> None:
    """Poll SQLite and push updates to the WebSocket client."""
    # Send init message
    metrics = await asyncio.to_thread(read_metrics_since, db_path, 0, MAX_METRICS_IN_INIT)
    games = await asyncio.to_thread(read_game_snapshots, db_path)
    state = await asyncio.to_thread(read_training_state, db_path)

    last_metrics_id = metrics[-1]["id"] if metrics else 0

    await asyncio.wait_for(
        ws.send_json({
            "type": "init",
            "games": games,
            "metrics": metrics,
            "training_state": state,
        }),
        timeout=WS_SEND_TIMEOUT_S,
    )

    # Poll loop
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)

        new_metrics = await asyncio.to_thread(
            read_metrics_since, db_path, last_metrics_id, 100
        )
        if new_metrics:
            last_metrics_id = new_metrics[-1]["id"]
            await asyncio.wait_for(
                ws.send_json({"type": "metrics_update", "rows": new_metrics}),
                timeout=WS_SEND_TIMEOUT_S,
            )

        new_games = await asyncio.to_thread(read_game_snapshots, db_path)
        if new_games:
            await asyncio.wait_for(
                ws.send_json({"type": "game_update", "snapshots": new_games}),
                timeout=WS_SEND_TIMEOUT_S,
            )

        new_state = await asyncio.to_thread(read_training_state, db_path)
        if new_state and state and (
            new_state.get("current_epoch") != state.get("current_epoch")
            or new_state.get("status") != state.get("status")
        ):
            state = new_state
            await asyncio.wait_for(
                ws.send_json({
                    "type": "training_status",
                    "status": new_state.get("status"),
                    "heartbeat_at": new_state.get("heartbeat_at"),
                    "epoch": new_state.get("current_epoch"),
                }),
                timeout=WS_SEND_TIMEOUT_S,
            )


async def _keepalive(ws: WebSocket) -> None:
    """Ping/pong heartbeat to detect dead connections."""
    while True:
        await asyncio.sleep(WS_PING_INTERVAL_S)
        try:
            await asyncio.wait_for(ws.send_json({"type": "ping"}), timeout=WS_SEND_TIMEOUT_S)
        except Exception:
            return


def main() -> None:
    """CLI entry point: keisei-serve."""
    import argparse
    import uvicorn

    from keisei.config import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Keisei spectator dashboard")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    config = load_config(args.config)
    app = create_app(config.display.db_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add keisei/server/__init__.py keisei/server/app.py tests/test_server.py
git commit -m "feat: FastAPI server with healthz, WebSocket polling, and keepalive"
```

---

### Task 2: WebSocket Protocol Tests

**Files:**
- Modify: `tests/test_server.py`

- [ ] **Step 1: Add WebSocket init test**

Append to `tests/test_server.py`:
```python
from httpx_ws import aconnect_ws
from keisei.db import write_metrics, write_game_snapshots


@pytest.mark.asyncio
async def test_ws_sends_init_on_connect(db_path: str) -> None:
    write_metrics(db_path, {"epoch": 0, "step": 100, "policy_loss": 1.5})
    app = create_app(db_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with aconnect_ws("/ws", client) as ws:
            msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
            assert msg["type"] == "init"
            assert "games" in msg
            assert "metrics" in msg
            assert "training_state" in msg
            assert msg["training_state"]["display_name"] == "TestBot"
```

Note: This test requires `httpx-ws` package. Add to dev deps if not present:
```bash
uv pip install httpx-ws
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_server.py::test_ws_sends_init_on_connect -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_server.py
git commit -m "test: WebSocket init message conformance test"
```
