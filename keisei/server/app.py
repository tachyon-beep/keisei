"""FastAPI spectator dashboard server."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from keisei.db import (
    _connect,
    init_db,
    read_elo_history,
    read_game_snapshots,
    read_game_snapshots_since,
    read_league_data,
    read_metrics_since,
    read_style_profiles,
    read_tournament_stats,
    read_training_state,
)
from keisei.showcase.db_ops import (
    queue_match as showcase_queue_match,
    read_queue as showcase_read_queue,
    read_active_showcase_game,
    read_all_showcase_moves,
    read_showcase_moves_since,
    read_heartbeat as showcase_read_heartbeat,
    cancel_match as showcase_cancel_match,
    update_queue_speed as showcase_update_speed,
)

logger = logging.getLogger(__name__)


def _style_fingerprint(profiles: list[dict[str, Any]]) -> tuple[tuple[int, str, str], ...]:
    """Lightweight fingerprint for style profiles to detect changes.

    Returns a tuple of (checkpoint_id, status, primary_style) per profile,
    which changes whenever profiles are recomputed.
    """
    return tuple(
        (p.get("checkpoint_id", 0), p.get("profile_status", ""), p.get("primary_style") or "")
        for p in profiles
    )


MAX_METRICS_IN_INIT = 500
POLL_INTERVAL_S = 0.2
ALLOWED_HOSTS = frozenset({"keisei.foundryside.dev", "192.168.1.240", "127.0.0.1", "localhost"})
# Superset for use in tests — includes synthetic hostnames from test clients
TEST_ALLOWED_HOSTS = ALLOWED_HOSTS | {"testserver", "test"}
SHOWCASE_POLL_INTERVAL_S = 0.5
VALID_SPEEDS = frozenset({"slow", "normal", "fast"})
MAX_SHOWCASE_QUEUE_DEPTH = 5
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


def _get_system_stats() -> dict[str, Any]:
    """Get CPU and GPU utilization stats."""
    stats: dict[str, Any] = {}
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
            gpus: list[dict[str, int]] = []
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


def _extract_hostname(host: str) -> str:
    """Extract hostname from a Host header, handling IPv6 bracketed literals.

    Examples: "localhost:8741" → "localhost", "[::1]:8741" → "::1", "" → ""
    """
    if host.startswith("["):
        # RFC 2732 bracketed IPv6: [::1]:port or [::1]
        bracket_end = host.find("]")
        if bracket_end != -1:
            return host[1:bracket_end]
        return host  # malformed — return as-is for rejection
    # IPv4 / hostname — strip port suffix
    return host.split(":")[0]


class HostFilterMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Host header isn't in the allowed set."""

    def __init__(self, app: Any, hosts: frozenset[str] = ALLOWED_HOSTS) -> None:
        super().__init__(app)
        self._hosts = hosts

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        host = request.headers.get("host", "")
        hostname = _extract_hostname(host)
        if hostname not in self._hosts:
            logger.warning("Rejected request with Host: %s", host)
            return PlainTextResponse("Forbidden", status_code=403)
        response: Response = await call_next(request)
        return response


def create_app(db_path: str, allowed_hosts: frozenset[str] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Server starting, db_path=%s", db_path)
        # Apply any pending schema migrations so dashboard queries
        # don't crash on columns added since the DB was created.
        await asyncio.to_thread(init_db, db_path)
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

    def _check_ws_host(websocket: WebSocket) -> bool:
        """Check websocket Host header against the allowlist.

        BaseHTTPMiddleware only filters HTTP scopes, not WebSocket scopes,
        so we enforce the same host allowlist here.
        """
        host = websocket.headers.get("host", "")
        hostname = _extract_hostname(host)
        return hostname in hosts

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        if not _check_ws_host(websocket):
            logger.warning("Rejected WebSocket with Host: %s", websocket.headers.get("host", ""))
            await websocket.close(code=1008, reason="Forbidden")
            return
        await websocket.accept()
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(_poll_and_push(websocket, db_path))
                tg.create_task(_keepalive(websocket))
                tg.create_task(_receive_commands(websocket, db_path))
                tg.create_task(_poll_showcase(websocket, db_path))
        except* WebSocketDisconnect:
            pass
        except* asyncio.CancelledError:
            # Normal: client disconnected while a background DB thread was
            # in flight.  CancelledError is a BaseException and escapes
            # except* Exception, so it needs its own clause.
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
    # Composite cursor for incremental game snapshot polling.
    # Tracks (timestamp, game_id) to avoid missing rows with equal timestamps.
    last_game_ts = ""
    last_game_id = 0
    if games:
        last_game_ts = max(g["updated_at"] for g in games)
        last_game_id = max(
            g["game_id"] for g in games if g["updated_at"] == last_game_ts
        )

    league_data = await asyncio.to_thread(read_league_data, db_path)
    elo_history = await asyncio.to_thread(read_elo_history, db_path)
    t_stats = await asyncio.to_thread(read_tournament_stats, db_path)
    style_profiles = await asyncio.to_thread(read_style_profiles, db_path)

    # Showcase init data
    showcase_game = await asyncio.to_thread(read_active_showcase_game, db_path)
    showcase_moves: list[dict[str, Any]] = []
    if showcase_game:
        showcase_moves = await asyncio.to_thread(read_all_showcase_moves, db_path, showcase_game["id"])
    showcase_queue = await asyncio.to_thread(showcase_read_queue, db_path)
    showcase_hb = await asyncio.to_thread(showcase_read_heartbeat, db_path)
    showcase_alive = False
    if showcase_hb:
        try:
            last_hb = datetime.fromisoformat(showcase_hb["last_heartbeat"].replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - last_hb).total_seconds()
            showcase_alive = age < HEARTBEAT_STALE_S
        except (ValueError, TypeError):
            pass

    await asyncio.wait_for(
        ws.send_json({
            "type": "init",
            "games": games,
            "metrics": metrics,
            "training_state": state,
            "league_entries": league_data["entries"],
            "league_results": league_data["results"],
            "historical_library": league_data["historical_library"],
            "gauntlet_results": league_data["gauntlet_results"],
            "transitions": league_data["transitions"],
            "elo_history": elo_history,
            "tournament_stats": t_stats,
            "style_profiles": style_profiles,
            "showcase": {
                "game": dict(showcase_game) if showcase_game else None,
                "moves": showcase_moves,
                "queue": showcase_queue,
                "sidecar_alive": showcase_alive,
            },
        }),
        timeout=WS_SEND_TIMEOUT_S,
    )

    last_league_entry_ids = frozenset(e["id"] for e in league_data["entries"])
    last_league_result_id = league_data["results"][0]["id"] if league_data["results"] else 0
    last_league_transition_id = league_data["transitions"][0]["id"] if league_data["transitions"] else 0
    # Track style profile state to avoid re-sending unchanged data.
    # Style profiles only change every ~5 tournament rounds, so a simple
    # fingerprint avoids redundant reads and sends on every poll tick.
    last_style_fingerprint = _style_fingerprint(style_profiles)
    league_poll_elapsed = 0.0
    total_episodes = sum((m.get("episodes_completed") or 0) for m in metrics)

    # Poll loop
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)

        new_metrics = await asyncio.to_thread(
            read_metrics_since, db_path, last_metrics_id, POLL_BATCH_SIZE
        )
        if new_metrics:
            last_metrics_id = new_metrics[-1]["id"]
            total_episodes += sum(
                (m.get("episodes_completed") or 0) for m in new_metrics
            )
            await asyncio.wait_for(
                ws.send_json({"type": "metrics_update", "rows": new_metrics}),
                timeout=WS_SEND_TIMEOUT_S,
            )

        changed_games, new_game_ts, new_game_id = await asyncio.to_thread(
            read_game_snapshots_since, db_path, last_game_ts, last_game_id
        )
        if changed_games:
            last_game_ts = new_game_ts
            last_game_id = new_game_id
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
                    "learner_entry_id": new_state.get("learner_entry_id"),
                }),
                timeout=WS_SEND_TIMEOUT_S,
            )

        league_poll_elapsed += POLL_INTERVAL_S
        if league_poll_elapsed >= LEAGUE_POLL_INTERVAL_S:
            league_poll_elapsed = 0.0
            new_league = await asyncio.to_thread(read_league_data, db_path)
            new_elo_hist = await asyncio.to_thread(read_elo_history, db_path)
            new_t_stats = await asyncio.to_thread(read_tournament_stats, db_path)
            new_entry_ids = frozenset(e["id"] for e in new_league["entries"])
            new_result_id = new_league["results"][0]["id"] if new_league["results"] else 0
            new_transition_id = new_league["transitions"][0]["id"] if new_league["transitions"] else 0
            league_changed = (
                new_entry_ids != last_league_entry_ids
                or new_result_id != last_league_result_id
                or new_transition_id != last_league_transition_id
            )
            # Only re-read style profiles when league data changed
            if league_changed:
                new_style = await asyncio.to_thread(read_style_profiles, db_path)
                new_fp = _style_fingerprint(new_style)
                style_changed = new_fp != last_style_fingerprint
                if style_changed:
                    last_style_fingerprint = new_fp
                    style_profiles = new_style
            else:
                style_changed = False

            if league_changed:
                last_league_entry_ids = new_entry_ids
                last_league_result_id = new_result_id
                last_league_transition_id = new_transition_id
                msg: dict[str, Any] = {
                    "type": "league_update",
                    "entries": new_league["entries"],
                    "results": new_league["results"],
                    "historical_library": new_league["historical_library"],
                    "gauntlet_results": new_league["gauntlet_results"],
                    "transitions": new_league["transitions"],
                    "elo_history": new_elo_hist,
                    "tournament_stats": new_t_stats,
                }
                if style_changed:
                    msg["style_profiles"] = style_profiles
                await asyncio.wait_for(
                    ws.send_json(msg),
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


async def _receive_commands(ws: WebSocket, db_path: str) -> None:
    """Listen for client-to-server commands on the WebSocket."""
    while True:
        try:
            raw = await ws.receive_text()
        except WebSocketDisconnect:
            raise
        except asyncio.CancelledError:
            # Scope cancellation during receive — treat as disconnect
            raise WebSocketDisconnect()
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Received non-JSON message from client")
            continue

        msg_type = data.get("type", "")
        try:
            if msg_type == "request_showcase_match":
                await _handle_match_request(ws, db_path, data)
            elif msg_type == "change_showcase_speed":
                await _handle_speed_change(ws, db_path, data)
            elif msg_type == "cancel_showcase_match":
                await _handle_cancel(ws, db_path, data)
            elif msg_type == "pong":
                pass  # client keepalive response
            else:
                logger.debug("Unknown client message type: %s", msg_type)
        except Exception:
            logger.exception("Error handling client command %s", msg_type)


async def _handle_match_request(ws: WebSocket, db_path: str, data: dict[str, Any]) -> None:
    """Validate and queue a showcase match request."""
    entry_id_1 = str(data.get("entry_id_1", ""))
    entry_id_2 = str(data.get("entry_id_2", ""))
    speed = data.get("speed", "normal")

    if speed not in VALID_SPEEDS:
        await asyncio.wait_for(
            ws.send_json({"type": "showcase_error", "error": f"Invalid speed: {speed}. Valid: {sorted(VALID_SPEEDS)}"}),
            timeout=WS_SEND_TIMEOUT_S,
        )
        return

    if not entry_id_1 or not entry_id_2:
        await asyncio.wait_for(
            ws.send_json({"type": "showcase_error", "error": "Both entry_id_1 and entry_id_2 are required"}),
            timeout=WS_SEND_TIMEOUT_S,
        )
        return

    if entry_id_1 == entry_id_2:
        await asyncio.wait_for(
            ws.send_json({"type": "showcase_error", "error": "Cannot match an entry against itself"}),
            timeout=WS_SEND_TIMEOUT_S,
        )
        return

    # Check queue depth
    queue = await asyncio.to_thread(showcase_read_queue, db_path)
    pending = [q for q in queue if q["status"] == "pending"]
    if len(pending) >= MAX_SHOWCASE_QUEUE_DEPTH:
        await asyncio.wait_for(
            ws.send_json({"type": "showcase_error", "error": "Queue is full"}),
            timeout=WS_SEND_TIMEOUT_S,
        )
        return

    await asyncio.to_thread(showcase_queue_match, db_path, entry_id_1, entry_id_2, speed)
    await asyncio.wait_for(
        ws.send_json({"type": "showcase_match_queued", "entry_id_1": entry_id_1, "entry_id_2": entry_id_2, "speed": speed}),
        timeout=WS_SEND_TIMEOUT_S,
    )


async def _handle_speed_change(ws: WebSocket, db_path: str, data: dict[str, Any]) -> None:
    """Change the speed of a queued/running match."""
    queue_id = data.get("queue_id")
    speed = data.get("speed", "")

    if speed not in VALID_SPEEDS:
        await asyncio.wait_for(
            ws.send_json({"type": "showcase_error", "error": f"Invalid speed: {speed}"}),
            timeout=WS_SEND_TIMEOUT_S,
        )
        return

    if queue_id is None:
        await asyncio.wait_for(
            ws.send_json({"type": "showcase_error", "error": "queue_id is required"}),
            timeout=WS_SEND_TIMEOUT_S,
        )
        return

    await asyncio.to_thread(showcase_update_speed, db_path, int(queue_id), speed)
    await asyncio.wait_for(
        ws.send_json({"type": "showcase_speed_changed", "queue_id": queue_id, "speed": speed}),
        timeout=WS_SEND_TIMEOUT_S,
    )


async def _handle_cancel(ws: WebSocket, db_path: str, data: dict[str, Any]) -> None:
    """Cancel a pending showcase match."""
    queue_id = data.get("queue_id")

    if queue_id is None:
        await asyncio.wait_for(
            ws.send_json({"type": "showcase_error", "error": "queue_id is required"}),
            timeout=WS_SEND_TIMEOUT_S,
        )
        return

    await asyncio.to_thread(showcase_cancel_match, db_path, int(queue_id))
    await asyncio.wait_for(
        ws.send_json({"type": "showcase_match_cancelled", "queue_id": queue_id}),
        timeout=WS_SEND_TIMEOUT_S,
    )


async def _poll_showcase(ws: WebSocket, db_path: str) -> None:
    """Poll showcase tables and push incremental updates.

    Uses incremental move delivery: only moves since last_sent_ply are sent.
    Status updates use fingerprinting to avoid redundant sends.
    """
    last_status_fingerprint: tuple[int | None, int, bool] = (None, 0, False)
    last_game_id: int | None = None
    last_sent_ply = 0

    while True:
        await asyncio.sleep(SHOWCASE_POLL_INTERVAL_S)

        game = await asyncio.to_thread(read_active_showcase_game, db_path)
        queue = await asyncio.to_thread(showcase_read_queue, db_path)
        hb = await asyncio.to_thread(showcase_read_heartbeat, db_path)

        alive = False
        if hb:
            try:
                last_hb = datetime.fromisoformat(hb["last_heartbeat"].replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - last_hb).total_seconds()
                alive = age < HEARTBEAT_STALE_S
            except (ValueError, TypeError):
                pass

        game_id = game["id"] if game else None

        # Reset cursor when game changes
        if game_id != last_game_id:
            last_sent_ply = 0
            last_game_id = game_id

        # Status fingerprint — only send when queue/game/sidecar state changes
        status_fingerprint = (game_id, len(queue), alive)
        if status_fingerprint != last_status_fingerprint:
            last_status_fingerprint = status_fingerprint
            try:
                await asyncio.wait_for(
                    ws.send_json({
                        "type": "showcase_status",
                        "queue": queue,
                        "active_game_id": game_id,
                        "sidecar_alive": alive,
                    }),
                    timeout=WS_SEND_TIMEOUT_S,
                )
            except (WebSocketDisconnect, ConnectionError, asyncio.TimeoutError):
                raise WebSocketDisconnect()

        # Send incremental moves only (not full history)
        if game:
            new_moves = await asyncio.to_thread(
                read_showcase_moves_since, db_path, game["id"], last_sent_ply,
            )
            if new_moves:
                last_sent_ply = max(m["ply"] for m in new_moves)
                try:
                    await asyncio.wait_for(
                        ws.send_json({
                            "type": "showcase_update",
                            "game": dict(game),
                            "new_moves": new_moves,
                        }),
                        timeout=WS_SEND_TIMEOUT_S,
                    )
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
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (ignored if --socket)")
    parser.add_argument("--port", type=int, default=8741, help="Bind port (ignored if --socket)")
    parser.add_argument("--socket", default=None, help="Unix domain socket path (overrides --host/--port)")
    args = parser.parse_args()

    config = load_config(args.config)
    app = create_app(config.display.db_path)
    if args.socket:
        uvicorn.run(app, uds=args.socket)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
