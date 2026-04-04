"""DemonstratorRunner — threaded inference-only exhibition matches."""

from __future__ import annotations

import logging
import random
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from keisei.training.league import OpponentEntry, OpponentPool

logger = logging.getLogger(__name__)


def _get_policy_flat(model_output: Any, batch_size: int) -> torch.Tensor:
    """Extract flat policy logits from either BaseModel or KataGoBaseModel output.

    BaseModel.forward() returns (policy_logits, value) — policy already flat.
    KataGoBaseModel.forward() returns KataGoOutput — policy is (B, 9, 9, 139).
    """
    if isinstance(model_output, tuple):
        result: torch.Tensor = model_output[0]
        return result
    else:
        result = model_output.policy_logits.reshape(batch_size, -1)
        return result


@dataclass
class DemoMatchup:
    """A demonstrator game matchup."""

    slot: int
    entry_a: OpponentEntry
    entry_b: OpponentEntry


class DemonstratorRunner(threading.Thread):
    """Runs inference-only exhibition matches in a background thread.

    Error policy: crashes are non-fatal. The run() method wraps the loop in
    try/except, logs the exception, and stops. The training loop can check
    is_alive() at epoch boundaries.
    """

    def __init__(
        self,
        pool: OpponentPool,
        db_path: str,
        num_slots: int = 3,
        moves_per_minute: int = 60,
        device: str = "cpu",
    ) -> None:
        super().__init__(daemon=True, name="DemonstratorRunner")
        self.pool = pool
        self.db_path = db_path
        self.num_slots = num_slots
        self.move_delay = 60.0 / max(moves_per_minute, 1)
        self.device = device
        self._stop_event = threading.Event()

        self._stream = None
        if device.startswith("cuda") and torch.cuda.is_available():
            self._stream = torch.cuda.Stream()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        """Thread main loop — wrapped in try/except for non-fatal crash policy."""
        try:
            self._run_loop()
        except Exception:
            logger.exception("DemonstratorRunner crashed — thread stopping")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            matchups = self._select_matchups()
            if not matchups:
                self._stop_event.wait(timeout=5.0)
                continue

            for matchup in matchups:
                if self._stop_event.is_set():
                    return
                try:
                    self._play_game(matchup)
                except Exception:
                    logger.exception("Demo slot %d game failed — skipping", matchup.slot)

    def _select_matchups(self) -> list[DemoMatchup]:
        """Select matchups for active demo slots based on pool state."""
        entries = self.pool.list_entries()
        if len(entries) < 2:
            return []

        matchups = []
        sorted_by_elo = sorted(entries, key=lambda e: e.elo_rating, reverse=True)

        # Slot 1: #1 Elo vs #2 Elo (championship match)
        if len(sorted_by_elo) >= 2 and self.num_slots >= 1:
            matchups.append(DemoMatchup(
                slot=1, entry_a=sorted_by_elo[0], entry_b=sorted_by_elo[1],
            ))

        # Slot 2: cross-architecture if available, else random
        if self.num_slots >= 2 and len(entries) >= 2:
            archs: dict[str, list[OpponentEntry]] = {}
            for e in entries:
                archs.setdefault(e.architecture, []).append(e)
            if len(archs) >= 2:
                arch_names = list(archs.keys())
                a = random.choice(archs[arch_names[0]])
                b = random.choice(archs[arch_names[1]])
                matchups.append(DemoMatchup(slot=2, entry_a=a, entry_b=b))
            else:
                pair = random.sample(entries, 2)
                matchups.append(DemoMatchup(slot=2, entry_a=pair[0], entry_b=pair[1]))

        # Slot 3: random pairing
        if self.num_slots >= 3 and len(entries) >= 2:
            pair = random.sample(entries, 2)
            matchups.append(DemoMatchup(slot=3, entry_a=pair[0], entry_b=pair[1]))

        return matchups

    def _play_game(self, matchup: DemoMatchup) -> None:
        """Play a single demonstrator game to completion."""
        self.pool.pin(matchup.entry_a.id)
        self.pool.pin(matchup.entry_b.id)
        try:
            model_a = self.pool.load_opponent(matchup.entry_a, device=self.device)
            model_b = self.pool.load_opponent(matchup.entry_b, device=self.device)
        except FileNotFoundError:
            logger.warning("Checkpoint missing for demo slot %d — skipping", matchup.slot)
            return
        finally:
            self.pool.unpin(matchup.entry_a.id)
            self.pool.unpin(matchup.entry_b.id)

        logger.info(
            "Demo slot %d: %s (elo=%.0f) vs %s (elo=%.0f)",
            matchup.slot,
            matchup.entry_a.architecture, matchup.entry_a.elo_rating,
            matchup.entry_b.architecture, matchup.entry_b.elo_rating,
        )

        try:
            from shogi_gym import VecEnv
            env = VecEnv(
                num_envs=1, max_ply=512,
                observation_mode="katago",  # type: ignore[call-arg]
                action_mode="spatial",
            )
        except ImportError:
            logger.warning("shogi_gym not available — demo slot %d inactive", matchup.slot)
            return

        reset_result = env.reset()
        obs = torch.from_numpy(np.asarray(reset_result.observations)).to(self.device)
        legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(self.device)
        current_player = 0
        models = [model_a, model_b]
        done = False

        while not done and not self._stop_event.is_set():
            model = models[current_player]
            ctx = torch.cuda.stream(self._stream) if self._stream else nullcontext()
            with ctx:
                with torch.no_grad():
                    # Guard: zero legal actions → all-inf softmax → NaN crash
                    legal_counts = legal_masks.sum(dim=-1)
                    if (legal_counts == 0).any():
                        logger.warning(
                            "Demo slot %d: zero legal actions detected — ending game",
                            matchup.slot,
                        )
                        break

                    output = model(obs)
                    flat = _get_policy_flat(output, obs.shape[0])
                    masked = flat.masked_fill(~legal_masks, float("-inf"))
                    probs = F.softmax(masked, dim=-1)
                    action = torch.distributions.Categorical(probs).sample()

            step_result = env.step(action.tolist())
            done = bool(step_result.terminated[0] or step_result.truncated[0])
            obs = torch.from_numpy(np.asarray(step_result.observations)).to(self.device)
            legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks)).to(self.device)
            current_player = int(step_result.current_players[0])
            time.sleep(self.move_delay)

        logger.info("Demo slot %d game completed", matchup.slot)
