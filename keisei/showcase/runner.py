"""Showcase sidecar: plays model-vs-model games at watchable speed.

Usage:
    python -m keisei.showcase.runner --db-path path/to/db.sqlite \\
        [--cpu-threads 2] [--auto-showcase-interval 1800] [--no-auto-showcase]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

import torch

from keisei.showcase.db_ops import (
    claim_next_match,
    cleanup_orphaned_games,
    create_showcase_game,
    mark_game_abandoned,
    mark_game_completed,
    queue_match,
    read_queue,
    update_queue_speed,
    write_heartbeat,
    write_showcase_move,
)
from keisei.showcase.inference import (
    ModelCache,
    enforce_cpu_only,
    load_model_for_showcase,
    run_inference,
)

MAX_PLY = 512
SPEED_DELAYS = {"slow": 4.0, "normal": 2.0, "fast": 0.5}
HEARTBEAT_INTERVAL = 10.0
POLL_INTERVAL = 5.0


class ShowcaseRunner:
    """Main sidecar runner for showcase games."""

    def __init__(
        self,
        db_path: str,
        cpu_threads: int = 2,
        auto_showcase_interval: int = 1800,
        auto_showcase_enabled: bool = True,
    ) -> None:
        self.db_path = db_path
        self.cpu_threads = cpu_threads
        self.auto_showcase_interval = auto_showcase_interval
        self.auto_showcase_enabled = auto_showcase_enabled
        self.model_cache = ModelCache(max_size=2)
        self._stop_event = threading.Event()
        self._speed_event = threading.Event()
        self._last_auto_showcase = 0.0

    def _startup_cleanup(self) -> None:
        count = cleanup_orphaned_games(self.db_path)
        if count > 0:
            logger.info("Cleaned up %d orphaned showcase game(s)", count)

    def _write_heartbeat(self) -> None:
        write_heartbeat(self.db_path, pid=os.getpid())

    def _get_delay(self, speed: str) -> float:
        return SPEED_DELAYS.get(speed, 2.0)

    def _create_env(self) -> Any:
        from shogi_gym import SpectatorEnv
        return SpectatorEnv(max_ply=MAX_PLY)

    def _load_models(self, match: dict[str, Any]) -> tuple[Any, Any, str, str]:
        from keisei.db import _connect
        conn = _connect(self.db_path)
        try:
            e1 = conn.execute("SELECT * FROM league_entries WHERE id = ?", (match["entry_id_1"],)).fetchone()
            e2 = conn.execute("SELECT * FROM league_entries WHERE id = ?", (match["entry_id_2"],)).fetchone()
        finally:
            conn.close()

        if e1 is None or e2 is None:
            raise ValueError(f"League entry not found: {match['entry_id_1']} or {match['entry_id_2']}")

        def _load_entry(entry: Any) -> tuple[Any, str]:
            arch = entry["architecture"]
            params_raw = entry["model_params"]
            params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
            model = self.model_cache.get_or_load(str(entry["id"]), entry["checkpoint_path"], arch, params)
            return model, arch

        model_black, arch_black = _load_entry(e1)
        model_white, arch_white = _load_entry(e2)
        return model_black, model_white, arch_black, arch_white

    def _run_game(self, match: dict[str, Any]) -> None:
        """Play a single showcase game. Queue entry ALWAYS finalized via finally."""
        game_id: int | None = None
        try:
            try:
                model_black, model_white, arch_black, arch_white = self._load_models(match)
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Cannot start showcase game: %s", e)
                return

            from keisei.db import _connect
            conn = _connect(self.db_path)
            try:
                e1 = conn.execute("SELECT * FROM league_entries WHERE id = ?", (match["entry_id_1"],)).fetchone()
                e2 = conn.execute("SELECT * FROM league_entries WHERE id = ?", (match["entry_id_2"],)).fetchone()
            finally:
                conn.close()

            game_id = create_showcase_game(
                self.db_path, queue_id=match["id"],
                entry_id_black=match["entry_id_1"], entry_id_white=match["entry_id_2"],
                elo_black=e1["elo_rating"] if e1 else 0.0, elo_white=e2["elo_rating"] if e2 else 0.0,
                name_black=e1["display_name"] if e1 else "Unknown", name_white=e2["display_name"] if e2 else "Unknown",
            )

            env = self._create_env()
            state = env.reset()
            logger.info("Showcase game %d started: %s vs %s", game_id,
                        e1["display_name"] if e1 else "?", e2["display_name"] if e2 else "?")

            ply = 0
            while not self._stop_event.is_set() and not env.is_over and ply < MAX_PLY:
                is_black_turn = state["current_player"] == "black"
                model = model_black if is_black_turn else model_white
                arch = arch_black if is_black_turn else arch_white

                obs = env.get_observation()
                start_ms = time.monotonic()
                policy_logits, win_prob = run_inference(model, obs, arch)
                inference_ms = int((time.monotonic() - start_ms) * 1000)

                legal = env.legal_actions()
                mask = np.full(policy_logits.shape, -1e9)
                mask[legal] = 0.0
                masked_logits = policy_logits + mask

                temperature = 0.5
                scaled_logits = masked_logits / temperature
                probs = np.exp(scaled_logits - scaled_logits.max())
                probs = probs / probs.sum()

                top_indices = np.argsort(probs)[::-1][:3]
                top_candidates = []
                for idx in top_indices:
                    if probs[idx] > 0.001:
                        top_candidates.append({"action": int(idx), "probability": round(float(probs[idx]), 4)})

                action = int(np.random.choice(len(probs), p=probs))

                state = env.step(action)
                ply = state["ply"]

                if state["move_history"]:
                    usi_notation = state["move_history"][-1]["notation"]
                else:
                    usi_notation = f"action_{action}"

                for tc in top_candidates:
                    tc["usi"] = usi_notation if tc["action"] == action else f"a{tc['action']}"

                write_showcase_move(
                    self.db_path, game_id=game_id, ply=ply, action_index=action,
                    usi_notation=usi_notation, board_json=json.dumps(state["board"]),
                    hands_json=json.dumps(state["hands"]), current_player=state["current_player"],
                    in_check=state.get("in_check", False), value_estimate=win_prob,
                    top_candidates=json.dumps(top_candidates), move_time_ms=inference_ms,
                )

                try:
                    from keisei.db import _connect as _db_connect
                    conn = _db_connect(self.db_path)
                    try:
                        row = conn.execute("SELECT speed FROM showcase_queue WHERE id = ?", (match["id"],)).fetchone()
                        speed = row["speed"] if row else match.get("speed", "normal")
                    finally:
                        conn.close()
                except Exception:
                    speed = match.get("speed", "normal")

                delay = self._get_delay(speed)
                self._speed_event.wait(timeout=delay)
                self._speed_event.clear()

            if self._stop_event.is_set():
                mark_game_abandoned(self.db_path, game_id, "shutdown")
                logger.info("Showcase game %d abandoned (shutdown)", game_id)
            elif ply >= MAX_PLY:
                mark_game_completed(self.db_path, game_id, "draw", total_ply=ply)
                logger.info("Showcase game %d ended: draw (max ply)", game_id)
            else:
                result = state.get("result", "in_progress")
                if result == "checkmate":
                    winner = "white" if state["current_player"] == "black" else "black"
                    status = f"{winner}_win"
                elif result in ("repetition", "perpetual_check", "impasse", "max_moves"):
                    status = "draw"
                else:
                    status = "draw"
                mark_game_completed(self.db_path, game_id, status, total_ply=ply)
                logger.info("Showcase game %d ended: %s (%d ply)", game_id, status, ply)

        except Exception:
            logger.exception("Showcase game failed (queue_id=%s, game_id=%s)", match["id"], game_id)
            if game_id is not None:
                try:
                    mark_game_abandoned(self.db_path, game_id, "error")
                except Exception:
                    logger.warning("Failed to abandon game %d during error recovery", game_id)

        finally:
            try:
                from keisei.db import _connect as _db_connect
                conn = _db_connect(self.db_path)
                try:
                    from keisei.showcase.db_ops import _now_iso
                    conn.execute(
                        "UPDATE showcase_queue SET status = 'completed', completed_at = ? WHERE id = ? AND status = 'running'",
                        (_now_iso(), match["id"]))
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                logger.warning("Failed to finalize queue entry %s", match["id"])

    def _maybe_auto_showcase(self) -> None:
        if not self.auto_showcase_enabled:
            return
        if time.monotonic() - self._last_auto_showcase < self.auto_showcase_interval:
            return
        queue = read_queue(self.db_path)
        if queue:
            return
        from keisei.db import _connect
        conn = _connect(self.db_path)
        try:
            rows = conn.execute("SELECT id FROM league_entries WHERE status = 'active' ORDER BY elo_rating DESC LIMIT 2").fetchall()
        finally:
            conn.close()
        if len(rows) < 2:
            return
        queue_match(self.db_path, str(rows[0]["id"]), str(rows[1]["id"]), "normal")
        self._last_auto_showcase = time.monotonic()
        logger.info("Auto-showcase: queued top-2 league entries")

    def run(self) -> None:
        enforce_cpu_only(self.cpu_threads)
        self._startup_cleanup()
        self._write_heartbeat()
        logger.info("Showcase runner started (pid=%d, db=%s)", os.getpid(), self.db_path)
        heartbeat_time = time.monotonic()

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - heartbeat_time >= HEARTBEAT_INTERVAL:
                self._write_heartbeat()
                heartbeat_time = now

            match = claim_next_match(self.db_path)
            if match is not None:
                try:
                    self._run_game(match)
                except Exception:
                    logger.exception("Unhandled error in _run_game (queue_id=%s)", match["id"])
                continue

            try:
                self._maybe_auto_showcase()
            except Exception:
                logger.warning("Auto-showcase check failed", exc_info=True)

            self._stop_event.wait(timeout=POLL_INTERVAL)

        logger.info("Showcase runner stopped")

    def stop(self) -> None:
        self._stop_event.set()
        self._speed_event.set()


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = argparse.ArgumentParser(description="Showcase sidecar runner")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--cpu-threads", type=int, default=2, help="PyTorch CPU threads")
    parser.add_argument("--auto-showcase-interval", type=int, default=1800, help="Seconds between auto-showcase matches")
    parser.add_argument("--no-auto-showcase", action="store_true", help="Disable auto-showcase")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    runner = ShowcaseRunner(
        db_path=args.db_path, cpu_threads=args.cpu_threads,
        auto_showcase_interval=args.auto_showcase_interval,
        auto_showcase_enabled=not args.no_auto_showcase,
    )

    def handle_signal(signum: int, frame: Any) -> None:
        logger.info("Received signal %d, stopping...", signum)
        runner.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    runner.run()


if __name__ == "__main__":
    main()
