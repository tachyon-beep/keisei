"""FastAPI spectator dashboard server."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from keisei.db import (
    read_game_snapshots,
    read_game_snapshots_since,
    read_league_data,
    read_elo_history,
    read_metrics_since,
    read_training_state,
)

logger = logging.getLogger(__name__)

MAX_METRICS_IN_INIT = 500
POLL_INTERVAL_S = 0.2
ALLOWED_HOSTS = frozenset({"keisei.foundryside.dev", "192.168.1.240", "127.0.0.1", "localhost"})
# Superset for use in tests — includes synthetic hostnames from test clients
TEST_ALLOWED_HOSTS = ALLOWED_HOSTS | {"testserver", "test"}
LEAGUE_POLL_INTERVAL_S = 5.0
POLL_BATCH_SIZE = 100
HEARTBEAT_STALE_S = 30
WS_SEND_TIMEOUT_S = 5.0
WS_PING_INTERVAL_S = 15.0


def _db_accessible(db_path: str) -> bool:
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.execute("SELECT 1 FROM schema_version")
            return True
        finally:
            conn.close()
    except Exception:
        return False


def _get_system_stats() -> dict:
    """Get CPU and GPU utilization stats."""
    import shutil
    stats = {}
    try:
        import psutil
        stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        stats["ram_used_gb"] = round(mem.used / (1024**3), 1)
        stats["ram_total_gb"] = round(mem.total / (1024**3), 1)
    except ImportError:
        stats["cpu_percent"] = None
        stats["ram_used_gb"] = None
        stats["ram_total_gb"] = None

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    gpus.append({
                        "util_percent": int(parts[0]),
                        "mem_used_mb": int(parts[1]),
                        "mem_total_mb": int(parts[2]),
                    })
            stats["gpus"] = gpus
    except Exception:
        stats["gpus"] = []

    return stats


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


class HostFilterMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Host header isn't in the allowed set."""

    def __init__(self, app, hosts: frozenset[str] = ALLOWED_HOSTS):
        super().__init__(app)
        self._hosts = hosts

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "")
        # Strip port from host header (e.g., "keisei.foundryside.dev:443" -> "keisei.foundryside.dev")
        hostname = host.split(":")[0]
        if hostname not in self._hosts:
            logger.warning("Rejected request with Host: %s", host)
            return PlainTextResponse("Forbidden", status_code=403)
        return await call_next(request)


def create_app(db_path: str, allowed_hosts: frozenset[str] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Server starting, db_path=%s", db_path)
        yield
        logger.info("Server shutting down")

    app = FastAPI(lifespan=lifespan)
    hosts = allowed_hosts if allowed_hosts is not None else ALLOWED_HOSTS
    app.add_middleware(HostFilterMiddleware, hosts=hosts)

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
    # Track the latest game snapshot timestamp for change detection.
    # Use the max updated_at from the initial fetch, or epoch zero if no games yet.
    last_game_ts = ""
    if games:
        last_game_ts = max(g["updated_at"] for g in games)

    league_data = await asyncio.to_thread(read_league_data, db_path)
    elo_history = await asyncio.to_thread(read_elo_history, db_path)

    await asyncio.wait_for(
        ws.send_json({
            "type": "init",
            "games": games,
            "metrics": metrics,
            "training_state": state,
            "league_entries": league_data["entries"],
            "league_results": league_data["results"],
            "elo_history": elo_history,
        }),
        timeout=WS_SEND_TIMEOUT_S,
    )

    last_league_entry_count = len(league_data["entries"])
    last_league_result_id = league_data["results"][0]["id"] if league_data["results"] else 0
    league_poll_elapsed = 0.0
    total_episodes = sum(m.get("episodes_completed", 0) for m in metrics)

    # Poll loop
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)

        new_metrics = await asyncio.to_thread(
            read_metrics_since, db_path, last_metrics_id, POLL_BATCH_SIZE
        )
        if new_metrics:
            last_metrics_id = new_metrics[-1]["id"]
            await asyncio.wait_for(
                ws.send_json({"type": "metrics_update", "rows": new_metrics}),
                timeout=WS_SEND_TIMEOUT_S,
            )

        changed_games, new_game_ts = await asyncio.to_thread(
            read_game_snapshots_since, db_path, last_game_ts
        )
        if changed_games:
            last_game_ts = new_game_ts
            await asyncio.wait_for(
                ws.send_json({"type": "game_update", "snapshots": changed_games}),
                timeout=WS_SEND_TIMEOUT_S,
            )

        new_state = await asyncio.to_thread(read_training_state, db_path)
        if new_state and (
            state is None
            or new_state.get("current_epoch") != state.get("current_epoch")
            or new_state.get("status") != state.get("status")
            or new_state.get("heartbeat_at") != (state or {}).get("heartbeat_at")
        ):
            sys_stats = await asyncio.to_thread(_get_system_stats)
            latest_metrics = await asyncio.to_thread(
                read_metrics_since, db_path, max(0, last_metrics_id - 1), 1
            )
            episodes = latest_metrics[-1].get("episodes_completed", 0) if latest_metrics else 0
            total_episodes += episodes
            state = new_state
            await asyncio.wait_for(
                ws.send_json({
                    "type": "training_status",
                    "status": new_state.get("status"),
                    "phase": new_state.get("phase", ""),
                    "heartbeat_at": new_state.get("heartbeat_at"),
                    "epoch": new_state.get("current_epoch"),
                    "step": new_state.get("current_step"),
                    "episodes": total_episodes,
                    "config_json": new_state.get("config_json"),
                    "display_name": new_state.get("display_name"),
                    "model_arch": new_state.get("model_arch"),
                    "total_epochs": new_state.get("total_epochs"),
                    "system_stats": sys_stats,
                }),
                timeout=WS_SEND_TIMEOUT_S,
            )

        league_poll_elapsed += POLL_INTERVAL_S
        if league_poll_elapsed >= LEAGUE_POLL_INTERVAL_S:
            league_poll_elapsed = 0.0
            new_league = await asyncio.to_thread(read_league_data, db_path)
            new_elo_hist = await asyncio.to_thread(read_elo_history, db_path)
            new_entry_count = len(new_league["entries"])
            new_result_id = new_league["results"][0]["id"] if new_league["results"] else 0
            if new_entry_count != last_league_entry_count or new_result_id != last_league_result_id:
                last_league_entry_count = new_entry_count
                last_league_result_id = new_result_id
                await asyncio.wait_for(
                    ws.send_json({
                        "type": "league_update",
                        "entries": new_league["entries"],
                        "results": new_league["results"],
                        "elo_history": new_elo_hist,
                    }),
                    timeout=WS_SEND_TIMEOUT_S,
                )


async def _keepalive(ws: WebSocket) -> None:
    """Ping/pong heartbeat to detect dead connections."""
    while True:
        await asyncio.sleep(WS_PING_INTERVAL_S)
        try:
            await asyncio.wait_for(ws.send_json({"type": "ping"}), timeout=WS_SEND_TIMEOUT_S)
        except (WebSocketDisconnect, ConnectionError, asyncio.TimeoutError):
            raise WebSocketDisconnect()


def create_app_from_env() -> FastAPI:
    """Factory for uvicorn --factory mode. Reads KEISEI_CONFIG env var."""
    import os

    from keisei.config import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config_path = os.environ.get("KEISEI_CONFIG", "keisei-league.toml")
    config = load_config(Path(config_path))
    return create_app(config.display.db_path)


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
    parser.add_argument("--port", type=int, default=8741, help="Bind port")
    args = parser.parse_args()

    config = load_config(args.config)
    app = create_app(config.display.db_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
