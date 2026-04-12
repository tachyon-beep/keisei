"""Tournament worker sidecar: claims pairings from the queue and plays them.

Usage:
    python -m keisei.training.tournament_runner \\
        --db-path data/keisei-500k-league.db \\
        --device cuda:1 \\
        --worker-id worker-0

The runner is a thin CLI wrapper around TournamentWorker, which is
unit-testable via injected stop_event and play_fn.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from typing import Any

import torch

from keisei.training.match_utils import MatchOutcome, play_match, release_models
from keisei.training.opponent_store import OpponentStore, compute_elo_update
from keisei.training.tournament import majority_wins_result
from keisei.training.tournament_queue import (
    claim_next_pairing,
    mark_pairing_done,
    reset_stale_playing,
    write_worker_heartbeat,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 2.0
HEARTBEAT_INTERVAL_SECONDS = 10.0


class TournamentWorker:
    """Main loop for a single tournament worker process."""

    def __init__(
        self, *,
        db_path: str,
        league_dir: str,
        worker_id: str,
        device: str = "cuda:1",
        games_per_match: int = 3,
        num_envs: int = 64,
        max_ply: int = 512,
        max_staleness_epochs: int = 50,
        k_factor: float = 16.0,
        stop_event: threading.Event | None = None,
        play_fn: Callable[..., MatchOutcome] | None = None,
        vecenv_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.db_path = db_path
        self.worker_id = worker_id
        self.device = torch.device(device)
        self.games_per_match = games_per_match
        self.num_envs = num_envs
        self.max_ply = max_ply
        self.max_staleness_epochs = max_staleness_epochs
        self.k_factor = k_factor
        self._stop_event = stop_event or threading.Event()
        self._play_fn = play_fn or play_match
        self._vecenv_factory = vecenv_factory
        self._pairings_done = 0
        self._last_heartbeat = 0.0

        self.store = OpponentStore(db_path=db_path, league_dir=league_dir)

    def _maybe_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
            write_worker_heartbeat(
                self.db_path, worker_id=self.worker_id,
                pid=os.getpid(), device=str(self.device),
                pairings_done=self._pairings_done,
            )
            self._last_heartbeat = now

    def _startup_sweep(self) -> None:
        """Reset any pairings this worker was playing when it last crashed."""
        n = reset_stale_playing(self.db_path, worker_id=self.worker_id)
        if n > 0:
            logger.info(
                "Startup sweep: reset %d stale 'playing' pairings for worker %s",
                n, self.worker_id,
            )

    def run(self) -> None:
        """Main loop: claim -> play -> record -> heartbeat -> repeat."""
        logger.info(
            "Tournament worker %s starting on %s", self.worker_id, self.device,
        )
        self._startup_sweep()
        self._maybe_heartbeat()

        if self._vecenv_factory is None:
            from shogi_gym import VecEnv

            def _default_factory() -> Any:
                return VecEnv(
                    num_envs=self.num_envs, max_ply=self.max_ply,
                    observation_mode="katago", action_mode="spatial",
                )

            self._vecenv_factory = _default_factory
        vecenv = self._vecenv_factory()

        try:
            while not self._stop_event.is_set():
                self._maybe_heartbeat()
                claim = claim_next_pairing(
                    self.db_path, worker_id=self.worker_id,
                )
                if claim is None:
                    self._stop_event.wait(POLL_INTERVAL_SECONDS)
                    continue
                current_epoch = self.store.get_current_epoch()
                if current_epoch - claim.enqueued_epoch > self.max_staleness_epochs:
                    logger.info(
                        "Expiring stale pairing %d (enqueued=%d, current=%d)",
                        claim.id, claim.enqueued_epoch, current_epoch,
                    )
                    mark_pairing_done(self.db_path, claim.id, status="expired")
                    continue
                try:
                    self._play_and_record(vecenv, claim, current_epoch)
                    mark_pairing_done(self.db_path, claim.id, status="done")
                    self._pairings_done += 1
                except Exception:
                    logger.exception(
                        "Failed pairing %d (%d vs %d)",
                        claim.id, claim.entry_a_id, claim.entry_b_id,
                    )
                    mark_pairing_done(self.db_path, claim.id, status="failed")
        finally:
            logger.info(
                "Tournament worker %s stopping (%d pairings completed)",
                self.worker_id, self._pairings_done,
            )

    def _play_and_record(self, vecenv: Any, claim: Any, epoch: int) -> None:
        """Load models, play match, record Elo + result."""
        entry_a = self.store.get_entry(claim.entry_a_id)
        entry_b = self.store.get_entry(claim.entry_b_id)
        if entry_a is None or entry_b is None:
            logger.warning(
                "Entry disappeared: a=%s b=%s", claim.entry_a_id, claim.entry_b_id,
            )
            return

        loaded: list[torch.nn.Module] = []
        try:
            model_a = self.store.load_opponent(entry_a, device=str(self.device))
            loaded.append(model_a)
            model_b = self.store.load_opponent(entry_b, device=str(self.device))
            loaded.append(model_b)

            outcome = self._play_fn(
                vecenv, model_a, model_b,
                device=self.device, num_envs=self.num_envs,
                max_ply=self.max_ply, games_target=claim.games_target,
                stop_event=self._stop_event,
            )
        finally:
            release_models(*loaded, device_type=self.device.type)

        total = outcome.a_wins + outcome.b_wins + outcome.draws
        if total == 0:
            return

        result_score = majority_wins_result(outcome.a_wins, outcome.b_wins, outcome.draws)
        new_a_elo, new_b_elo = compute_elo_update(
            entry_a.elo_rating, entry_b.elo_rating,
            result=result_score, k=self.k_factor,
        )
        self.store.record_result(
            epoch=epoch,
            entry_a_id=entry_a.id,
            entry_b_id=entry_b.id,
            wins_a=outcome.a_wins,
            wins_b=outcome.b_wins,
            draws=outcome.draws,
            match_type="calibration",
            role_a=entry_a.role,
            role_b=entry_b.role,
            elo_before_a=entry_a.elo_rating,
            elo_after_a=new_a_elo,
            elo_before_b=entry_b.elo_rating,
            elo_after_b=new_b_elo,
            training_updates_a=entry_a.update_count,
            training_updates_b=entry_b.update_count,
        )
        self.store.update_elo(entry_a.id, new_a_elo, epoch=epoch)
        self.store.update_elo(entry_b.id, new_b_elo, epoch=epoch)

        logger.info(
            "  %s vs %s — %dW %dL %dD (Elo: %.0f→%.0f / %.0f→%.0f)",
            entry_a.display_name, entry_b.display_name,
            outcome.a_wins, outcome.b_wins, outcome.draws,
            entry_a.elo_rating, new_a_elo,
            entry_b.elo_rating, new_b_elo,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tournament worker sidecar")
    p.add_argument("--db-path", required=True)
    p.add_argument("--league-dir", required=True)
    p.add_argument("--worker-id", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--games-per-match", type=int, default=3)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--max-ply", type=int, default=512)
    p.add_argument("--max-staleness-epochs", type=int, default=50)
    p.add_argument("--k-factor", type=float, default=16.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    stop_event = threading.Event()

    def _handle_signal(signum: int, frame: Any) -> None:
        logger.info("Received signal %d, stopping gracefully", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    worker = TournamentWorker(
        db_path=args.db_path,
        league_dir=args.league_dir,
        worker_id=args.worker_id,
        device=args.device,
        games_per_match=args.games_per_match,
        num_envs=args.num_envs,
        max_ply=args.max_ply,
        max_staleness_epochs=args.max_staleness_epochs,
        k_factor=args.k_factor,
        stop_event=stop_event,
    )
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
